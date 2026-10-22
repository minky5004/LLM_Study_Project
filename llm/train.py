"""학습 루프.

지금은 조각 5까지: 장치 · 모델 · 옵티마이저 준비 (문서 1-1 ~ 1-3) · 틀린 정도 계산 (문서 2-1 ~ 2-3)
· 한 번 고치기와 반복 (문서 3-1 ~ 3-4) · 학습용/검증용 손실 점검 (문서 4-1 ~ 4-3).
학습한 숫자 저장 · 불러와 확인 (문서 5-1 ~ 5-2).
8단계 조각 3: 모델 키우기(6층 · 256칸) · dropout 0.2 · 검증 손실이 가장 좋을 때만 저장 · 설정도 같이 저장 (8단계 문서 3-1 ~ 3-5).
9단계 조각 7: 토크나이저가 BPE 1024 로 바뀜 · 맨 아래에 글자당 손실 (9단계 문서 7-5).
  줄 사이 `# 결과:` 중 1244 · 5,410,012 같은 값은 8단계(글자 단위) 기록 · 9단계 값은 파일 끝 블록에.
"""

# asdict: dataclass(설정 상자)를 {이름: 값} 사전으로 바꾸는 함수. 모델 파일에 설정을 같이 넣을 때 쓴다 (8단계 문서 3-5).
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from llm.config import ModelConfig, TrainConfig
from llm.dataset import get_batch, train_data, val_data
from llm.model import GPT

# 주의: tokenizer 를 import 하면 그 모듈 최상위가 실행되어 train.txt 를 읽는다. 프로젝트 루트에서 `-m` 으로 실행해야 경로가 맞는다.
# vocab_size: 토크나이저가 글에서 센 글자 종류 수. 5단계에서 한 것처럼 모델 설정에 넘겨 config.py 의 손으로 적은 69 에 기대지 않는다.
from llm.tokenizer import train_text, val_text, vocab_size


# ---- 조각 1: 장치 · 모델 · 옵티마이저 준비 ----

# 계산을 할 장치를 고른다 (문서 1-1).
# GPU(NVIDIA) 를 쓸 수 있으면 "cuda" · 없으면 "cpu". 이 PC 는 RTX 3060 이라 "cuda" 가 나와야 한다.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 결과: device = "cuda" (RTX 3060 을 잡음)

# 설정 상자 두 개를 만든다. 모델 크기는 ModelConfig · 학습 방법은 TrainConfig 가 들고 있다.
# 글자 종류 수는 토크나이저가 센 값으로 덮어쓴다 (5단계 4-1 과 같다).
# 8단계 조각 3: 대화 1만 개 데이터에 맞춰 크기 · dropout · 학습 횟수를 바꾼다. 같은 조건에서 재 본 결과로 고른 값이다 (8단계 문서 3-2 · 3-3).
#   4층 · 128칸 · 3000번 → 검증 1.88 (덜 배움) · 6층 · 256칸 · dropout 0 → 검증 최저 1.72 뒤 외우기
#   6층 · 256칸 · dropout 0.2 · 5000번 → 검증 최저 1.63 (4500번째) · 외우기가 거의 없음
# config.py 의 기본값(4층 · 128칸 · 0.1)은 4단계 확인용 그대로 두고, 학습할 때만 여기서 덮어쓴다.
# 이 값은 모델 파일에 같이 저장되어 generate.py 가 그대로 읽는다 — 크기를 두 곳에 적지 않는다 (8단계 문서 3-5).
# n_head=8: 256 칸을 8 갈래로 쪼개 갈래마다 32칸 (지금까지와 같은 갈래 크기).
model_config = ModelConfig(vocab_size=vocab_size, n_layer=6, n_embd=256, n_head=8, dropout=0.2)
train_config = TrainConfig(max_iters=5000)

# 모델을 만들어 장치로 옮긴다 (문서 1-2).
# 모델의 숫자(파라미터)가 GPU 메모리에 있어야 GPU 가 계산할 수 있다. 한 번 옮기면 끝이다.
model = GPT(model_config).to(device)
# 결과: 모델 숫자가 cuda:0 (0번 GPU) 에 놓임 · 파라미터 5,410,012개 (6층 · 256칸 · 글자 1244개 — 7단계까지 4층 · 128칸 · 69개일 땐 827,461)

# 옵티마이저를 만든다 (문서 1-3).
# AdamW 에 "고칠 숫자 목록" 과 "한 번에 얼마나 고칠지(학습률)" 를 넘긴다. 아직 고치지는 않는다 — 고치는 일은 조각 3 에서 한다.
# lr 에는 TrainConfig 에 든 학습률을 꺼내 넘긴다 (3e-4).
optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
# 결과: optimizer.param_groups[0]["lr"] = 0.0003


# ---- 조각 2: 손실 계산 ----

# 묶음 하나를 뽑아 모델과 같은 장치로 옮긴다 (문서 1-2 ③).
# get_batch 는 CPU 텐서를 돌려주므로 여기서 옮긴다. .to() 는 새 텐서를 돌려주니 꼭 받아야 한다.
xb, yb = get_batch("train")
xb, yb = xb.to(device), yb.to(device)
# 결과: xb.shape = yb.shape = torch.Size([32, 128]) · xb.device = cuda:0

# 모델에 넣어 글자 자리마다 글자 종류 수(1244)만큼 점수를 받는다.
logits = model(xb)
# 결과: logits.shape = torch.Size([32, 128, 1244]) = (B, T, V) — 묶음 32 · 글자 자리 128 · 글자 종류 1244
B, T, V = logits.shape

# 점수와 정답을 "문제 B*T 개" 모양으로 편다 (문서 2-2).
# cross_entropy 는 점수 (N, V) · 정답 (N,) 을 기대한다. 줄 · 자리를 구분할 이유가 없으니 앞 두 축을 합친다.
logits_flat = logits.view(B * T, V)
# 결과: logits_flat.shape = torch.Size([4096, 1244]) — 문제 4096개 × 보기 1244개
targets_flat = yb.view(B * T)
# 결과: targets_flat.shape = torch.Size([4096]) — 문제 4096개의 정답

# 틀린 정도를 숫자 하나로 만든다 (문서 2-1).
# 점수는 확률로 바꾸기 전 그대로 넣는다. 확률 변환 · 정답 글자 확률 꺼내기 · 평균은 함수가 안에서 한다.
loss = F.cross_entropy(logits_flat, targets_flat)
# 결과: loss = 7.2626 (스칼라 텐서) — 학습 전이라 찍어서 맞힐 때의 ln(1244) ≈ 7.13 근처. 묶음이 무작위라 실행마다 조금 다르다 (문서 2-3)
#       (글자 69개일 땐 4.32 · ln(69) ≈ 4.23)


# ---- 조각 3: 한 번 고치기 → 반복 ----

# 한 번 고치기 (문서 3-1 ~ 3-3). 위 조각 2 에서 잰 loss 를 가지고 모델 숫자를 고친다.
# 순서는 "기울기 비우기 → 기울기 계산 → 숫자 고치기" 다.

# 옵티마이저가 들고 있는 이전 기울기 메모를 비운다 (문서 3-3).
# 안 비우면 계산한 기울기가 이전 것에 더해져 섞인다.
optimizer.zero_grad()

# loss 에서 거꾸로 거슬러 올라가 모든 파라미터의 기울기를 계산한다 (문서 3-1).
# 기울기 = "그 숫자를 올리면 loss 가 얼마나 변하나". 계산 결과는 각 파라미터 안에 저장된다.
loss.backward()

# 계산된 기울기를 보고 모든 파라미터를 학습률(3e-4)만큼 고친다 (문서 3-2).
optimizer.step()

# 같은 묶음을 다시 재면 손실이 내려가 있어야 한다. 고친 효과를 눈으로 보는 확인이다.
loss_after = F.cross_entropy(model(xb).view(B * T, V), targets_flat)
# 결과: loss 7.26 → loss_after 6.64 — 한 걸음만으로 내려감 (값은 묶음이 무작위라 실행마다 조금 다르다)


# ---- 조각 4: 점검 — 학습용 · 검증용 손실 평균 ----

# 이 함수 안에서는 기울기를 기록하지 않는다 (문서 4-3 ②). 데코레이터 = 함수 위에 붙여 동작을 바꾸는 표시.
# 점수만 재고 고치지 않으니 기록이 필요 없고, 안 하면 메모리 · 시간만 쓴다.
@torch.no_grad()
def estimate_loss():
    """학습용 · 검증용 글에서 각각 eval_iters 개 묶음의 손실을 평균 내어 {"train": ..., "val": ...} 로 돌려준다 (문서 4-1 · 4-2)."""
    result = {}

    # dropout 을 끄고 재도록 모델을 점검 모드로 바꾼다 (문서 4-3 ③). 매번 같은 조건으로 재려는 것이다.
    model.eval()

    for split in ("train", "val"):
        # 묶음마다 손실을 적어 둘 빈 칸 eval_iters(50) 개
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            # split 에 맞는 글에서 묶음을 뽑는다 — 첫 번째 반복은 "train" · 두 번째는 "val" (문서 4-2).
            xb, yb = get_batch(split)
            xb, yb = xb.to(device), yb.to(device)

            # 점수 → 펴기 → 손실. 조각 2 와 같고, 고치는 세 줄(zero_grad · backward · step)은 없다 — 점수만 재는 일이다.
            logits = model(xb)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), yb.view(B * T))
            # 숫자 하나짜리 텐서에서 파이썬 숫자만 꺼내 k 번째 칸에 적는다.
            losses[k] = loss.item()

        # 50개 손실을 평균 낸다 (문서 4-1 ②). 결과는 숫자 하나짜리 텐서이니 .item() 으로 꺼낸다.
        result[split] = losses.mean().item()
        # 결과: 학습 끝 무렵 result = {"train": 1.21.., "val": 1.64..} 꼴 — 아래 train() 출력 참고

    # 점검이 끝났으니 학습 모드로 되돌린다. 안 되돌리면 이후 학습이 dropout 없이 돈다 (문서 4-3 ④).
    model.train()
    return result


# ---- 조각 5: 학습한 모델 저장 ----

# 저장할 파일 경로. checkpoints/ 폴더는 .gitignore 에 있어 리포에 안 올라간다 (문서 5-1 ③).
checkpoint_path = Path("checkpoints/model.pt")


def save_model():
    """모델이 배운 숫자와 모델 설정을 함께 파일에 적는다 (문서 5-2 ① ② · 8단계 문서 3-5)."""
    # 폴더가 없으면 만든다. 이미 있어도 에러 없이 넘어간다 (exist_ok=True).
    checkpoint_path.parent.mkdir(exist_ok=True)

    # 모델의 "이름 → 숫자 덩어리" 사전을 뽑는다 (문서 5-2 ①).
    weights = model.state_dict()

    # 8단계 조각 3: 숫자만 적던 것을 {설정, 숫자} 사전 하나로 적는다.
    # 크기(6층 · 256칸)를 학습 때 바꾸므로, 불러오는 쪽이 그 크기를 이 파일에서 읽어 같은 모양의 빈 모델을 만든다 (8단계 문서 3-5).
    # asdict(model_config) = {"vocab_size": ..., "n_layer": 6, ...} — 숫자 · 글자만 든 사전이라 torch.load 가 안전하게 읽는다.
    # "model" 이름표에는 바로 위에서 뽑은 모델의 숫자 사전 weights.
    torch.save({"config": asdict(model_config), "model": weights}, checkpoint_path)
    # 결과: checkpoints/model.pt 21,668,891 바이트 (약 21MB = 5,410,012개 × 4바이트 + 설정) · 7단계까지는 숫자만 3.2MB
    #       torch.load 하면 키 ['config', 'model'] · config = {'vocab_size': 1244, 'block_size': 128, 'n_embd': 256, 'n_head': 8, 'n_layer': 6, 'dropout': 0.2}


def check_saved_model():
    """저장한 파일만으로 빈 모델을 만들어 불러오고, 학습한 모델과 같은 점수가 나오는지 본다 (문서 5-2 ③ ④)."""
    # 파일을 읽는다. 안에 {"config": 설정 사전, "model": 숫자 사전} 이 들어 있다 (map_location 은 GPU 로 바로 읽으라는 뜻).
    ckpt = torch.load(checkpoint_path, map_location=device)

    # 파일에 적힌 설정만으로 빈 모델을 만든다 — train.py 의 model_config 를 쓰지 않는다.
    # 그래야 "파일 하나가 자기 크기를 기억한다" 는 것까지 확인된다 (8단계 문서 3-5).
    # ModelConfig(**사전): 사전의 이름표를 인자 이름으로 풀어 넣는다. ModelConfig(vocab_size=1244, n_layer=6, ...) 와 같다.
    fresh = GPT(ModelConfig(**ckpt["config"])).to(device)

    # 파일의 숫자를 빈 모델에 채워 넣는다.
    fresh.load_state_dict(ckpt["model"])

    # dropout 이 켜져 있으면 점수가 매번 달라지니 둘 다 점검 모드로 바꾸고 같은 입력을 넣는다.
    model.eval()
    fresh.eval()
    with torch.no_grad():
        return torch.equal(model(xb), fresh(xb))
        # 결과: True — 파일만으로 만든 모델의 점수가 (가장 좋았을 때로 되돌린) 학습한 모델과 한 칸도 다르지 않다


# 9단계 마무리: 점검 때마다 손실을 여기 모아 두었다가 학습이 끝나면 파일로 남긴다 (아래 save_log).
# 화면 출력은 실행이 끝나면 사라져서, 나중에 실험끼리 그래프로 비교하려면 숫자가 파일에 있어야 한다.
history = []  # [{"step": 0, "train": 6.27, "val": 6.28}, ...]


def train():
    """묶음 뽑기 → 점수 → 손실 → 고치기를 max_iters 번 되풀이한다 (문서 3-4).

    8단계 조각 3: 점검 때마다 검증 손실이 지금까지의 최저보다 낮으면 그때만 저장한다 (8단계 문서 3-4).
    돌려주는 값: 가장 좋았던 검증 손실.
    """
    # 지금까지의 최저 검증 손실. 처음엔 무한대(어떤 숫자보다 큼) — 첫 점검은 무조건 "더 낮음" 이 되어 저장된다.
    best_val = float("inf")

    # 몇 번 되풀이할지는 TrainConfig 의 max_iters 가 정한다 (위에서 5000 으로 덮어씀).
    for step in range(train_config.max_iters):
        # 매번 새 묶음을 뽑아 장치로 옮긴다. 같은 묶음만 보면 그 묶음만 외우게 된다.
        xb, yb = get_batch("train")
        xb, yb = xb.to(device), yb.to(device)

        # 점수 → 펴기 → 손실. 조각 2 와 같다.
        logits = model(xb)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), yb.view(B * T))

        # 한 번 고치기. 위에서 한 세 줄과 같다.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 일정 간격마다 학습용 · 검증용 손실을 찍어 내려가는지 본다.
        # 찍는 간격은 eval_interval (300). 한 묶음 손실이 아니라 eval_iters (50) 개 묶음 평균이다 (조각 4).
        if step % train_config.eval_interval == 0:
            losses = estimate_loss()
            print(f"step {step}: train {losses['train']:.4f} · val {losses['val']:.4f}")
            history.append({"step": step, "train": losses["train"], "val": losses["val"]})  # 파일로 남길 한 줄

            # 8단계 조각 3: 시험 점수가 지금까지 중 가장 좋으면 그 순간의 모델을 저장한다.
            # 외우기가 시작돼 나중에 나빠져도 파일에는 가장 좋았던 모델이 남는다 (8단계 문서 3-4).
            # "지금까지의 최저" best_val 보다 낮을 때만.
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_model()
                print(f"  저장 (검증 최저 {best_val:.4f})")

    return best_val
            # 결과: step 900 → train 2.05 · val 2.11 / 1800 → 1.75 · 1.86 / 3000 → 1.51 · 1.73 / 4800 → 1.21 · 1.64 (묶음이 무작위라 실행마다 조금 다르다)
            #       검증이 매 점검마다 최저를 새로 써서 17번 모두 저장됨 · 마지막 저장 4800번째 검증 1.6435 · 혼자 돌려 249초
            #       (7단계까지 토이 데이터: 2700 → 0.18 · 0.19 — 46종류를 외운 점수라 비교 대상이 아니다)
            # 둘이 거의 같다 — 대화가 학습용 · 검증용에 다 들어 있는 토이 데이터라서 (5단계). 외운 게 아니라는 증거는 못 된다


def save_log(best_val, seconds):
    """이번 학습의 설정 · 점검마다의 손실 · 요약을 JSON 파일 하나로 남긴다 — 나중에 실험끼리 그래프로 비교하려고.

    파일: checkpoints/logs/<날짜-시각>-v<vocab_size>.json (리포 밖 — checkpoints/ 는 .gitignore)
    글자당 손실로 바꾸는 데 필요한 토큰 수 · 글자 수도 같이 적는다 (9단계 문서 7-5) — 그래프 때 다시 encode 하지 않게.
    """
    log_dir = checkpoint_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{time.strftime('%Y-%m-%d-%H%M')}-v{vocab_size}.json"
    # 어느 코드로 돌렸는지 — git 의 지금 커밋 번호 (git 이 없으면 빈칸)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    best = min(history, key=lambda h: h["val"])  # 검증 최저였던 점검 한 줄
    log = {
        "commit": commit,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "params": sum(p.numel() for p in model.parameters()),
        "train_tokens": len(train_data), "train_chars": len(train_text),
        "val_tokens": len(val_data), "val_chars": len(val_text),
        "seconds": round(seconds, 1),  # 학습 루프만 (import 때의 encode 는 빠짐)
        "best_step": best["step"], "best_val": best_val,
        "best_val_per_char": best_val * len(val_data) / len(val_text),
        "history": history,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)  # ensure_ascii=False: 한글을 \uXXXX 가 아니라 글자 그대로
    return path


# 직접 실행했을 때만 확인용 출력 (uv run python -m llm.train)
if __name__ == "__main__":
    print(device)  # 기대: cuda (GPU 를 못 잡으면 cpu — 그러면 CUDA 판 torch 설치를 먼저 본다)
    # 모델 첫 숫자 덩어리가 어느 장치에 있는지 — device 와 같아야 한다
    print(next(model.parameters()).device)  # 기대: cuda:0 (cuda 의 0번 GPU)
    # 모델 안의 파라미터 총 개수 — 4단계에서 센 값과 같아야 한다
    print(sum(p.numel() for p in model.parameters()))  # 기대: 5297152 (6층 · 256칸 · BPE 1024 — 8단계 글자 1244개일 땐 5410012)
    # 옵티마이저가 들고 있는 학습률 — TrainConfig 의 값과 같아야 한다
    print(optimizer.param_groups[0]["lr"])  # 기대: 0.0003
    # 틀린 정도 — 학습 전이라 1244글자를 찍어 맞히는 값 ln(1244) 근처여야 한다
    print(loss.item())  # 기대: ln(1024) ≈ 6.93 근처 (8단계 글자 1244개일 땐 ln(1244) ≈ 7.13 — 묶음이 무작위라 실행마다 조금 다르다)
    # 펴기 전 · 후 모양 — (B, T, V) 가 (B*T, V) 로
    print(logits.shape, logits_flat.shape, targets_flat.shape)  # 기대: [32,128,1024] [4096,1024] [4096] (8단계는 1244)
    # 한 번 고친 뒤 같은 묶음으로 다시 잰 손실 — 위 loss 보다 작아야 한다
    print(loss_after.item())  # 기대: loss 보다 작은 값 (한 걸음만 갔으니 조금만)
    # 반복 학습 — 점검마다 손실을 찍고, 검증 최저를 갱신할 때만 저장한다
    t = time.perf_counter()
    best_val = train()
    print(save_log(best_val, time.perf_counter() - t))  # 남긴 로그 파일 경로
    print(best_val)  # 가장 좋았던 검증 손실 — 결과: 2.301997423171997 (4500번째 · 토큰당) · 8단계 글자 단위 1.6435155868530273 (4800번째)

    # 9단계 조각 7: 토큰 크기가 다른 토크나이저끼리 비교할 글자당 손실 (9단계 문서 7-5).
    # 토큰당 손실 × 검증 토큰 수 = 검증용 글 전체를 맞힐 때 틀린 정도의 합 · 이걸 검증 글자 수로 나눈다.
    # 글자 단위 토크나이저면 토큰 수 = 글자 수라 best_val 과 똑같은 값이 나온다 — 8단계 1.6435 와 같은 잣대
    # 토큰당 손실 × 검증용 토큰 수(val_data 의 길이 — 텐서도 len 으로 첫 차원 길이를 센다) ÷ 검증용 글자 수
    per_char = best_val * len(val_data) / len(val_text)
    print(per_char)
    # 결과: 1.711378964294574 (= 2.3020 × 34,722 토큰 ÷ 46,705 글자) — 8단계 1.6435 보다 약 4% 나쁨
    #       토큰당 2.30 은 8단계 1.64 보다 훨씬 커 보이지만 같은 잣대로는 차이가 작다 (9단계 문서 7-5)

    # 학습이 끝난 model 은 "마지막" 숫자다. 파일에는 "가장 좋았던" 숫자가 있다 — 둘이 다를 수 있다.
    # 파일의 숫자를 model 로 되돌려 이후 확인 · 사용이 가장 좋았던 모델로 이뤄지게 한다.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])

    # 저장한 파일만으로 만든 모델과 점수를 비교한다
    print(check_saved_model())  # 기대: True (같은 점수 — 파일의 설정으로 만든 빈 모델에 숫자가 그대로 들어감)

# ---- 9단계 조각 7 · BPE 1024 로 다시 학습한 결과 (uv run python -m llm.train · 5분 35초 — 학습용 글 encode 약 1분 포함) ----
# 파라미터 5,297,152 · 학습 전 손실 7.03 → 한 걸음 뒤 6.63
# step 0: train 6.2744 · val 6.2804
# step 300: train 3.2430 · val 3.3074     <- 공통 잣대 300번째 손실 (8단계 글자 단위와는 토큰 크기가 달라 바로 비교 안 됨)
# step 900: train 2.7938 · val 2.9164
# step 1800: train 2.4192 · val 2.6303
# step 3000: train 1.9798 · val 2.4002
# step 4500: train 1.5616 · val 2.3020    <- 검증 최저 (저장)
# step 4800: train 1.4732 · val 2.3042    <- 검증은 멈췄는데 학습은 계속 내려감 — 외우기 시작
# 2.301997423171997 · 글자당 1.711378964294574 · True
# 학습 · 검증 차이 1.47 vs 2.30 (4800번째) — 8단계 1.21 vs 1.64 보다 벌어짐.
#   같은 5000번 × 32줄 × 128칸 = 토큰 2,048만 개를 봐도 토큰이 커서 글을 더 많이 본다 — 학습용 글 전체를 약 67바퀴 · 8단계는 약 50바퀴 (계산)
#   그래서 같은 횟수에 더 일찍 외우는 쪽으로 간다 (추정 — 횟수를 줄여 재지는 않음)
