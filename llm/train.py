"""학습 루프.

지금은 조각 5까지: 장치 · 모델 · 옵티마이저 준비 (문서 1-1 ~ 1-3) · 틀린 정도 계산 (문서 2-1 ~ 2-3)
· 한 번 고치기와 반복 (문서 3-1 ~ 3-4) · 학습용/검증용 손실 점검 (문서 4-1 ~ 4-3).
학습한 숫자 저장 · 불러와 확인 (문서 5-1 ~ 5-2).
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from llm.config import ModelConfig, TrainConfig
from llm.dataset import get_batch
from llm.model import GPT

# 주의: tokenizer 를 import 하면 그 모듈 최상위가 실행되어 train.txt 를 읽는다. 프로젝트 루트에서 `-m` 으로 실행해야 경로가 맞는다.
# vocab_size: 토크나이저가 글에서 센 글자 종류 수. 5단계에서 한 것처럼 모델 설정에 넘겨 config.py 의 손으로 적은 69 에 기대지 않는다.
from llm.tokenizer import vocab_size


# ---- 조각 1: 장치 · 모델 · 옵티마이저 준비 ----

# 계산을 할 장치를 고른다 (문서 1-1).
# GPU(NVIDIA) 를 쓸 수 있으면 "cuda" · 없으면 "cpu". 이 PC 는 RTX 3060 이라 "cuda" 가 나와야 한다.
device = "cuda" if torch.cuda.is_available() else "cpu"
# 결과: device = "cuda" (RTX 3060 을 잡음)

# 설정 상자 두 개를 만든다. 모델 크기는 ModelConfig · 학습 방법은 TrainConfig 가 들고 있다.
# 글자 종류 수는 토크나이저가 센 값으로 덮어쓴다 (5단계 4-1 과 같다).
model_config = ModelConfig(vocab_size=vocab_size)
train_config = TrainConfig()

# 모델을 만들어 장치로 옮긴다 (문서 1-2).
# 모델의 숫자(파라미터)가 GPU 메모리에 있어야 GPU 가 계산할 수 있다. 한 번 옮기면 끝이다.
model = GPT(model_config).to(device)
# 결과: 모델 숫자가 cuda:0 (0번 GPU) 에 놓임 · 파라미터 827461개

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

# 모델에 넣어 글자 자리마다 69개 점수를 받는다.
logits = model(xb)
# 결과: logits.shape = torch.Size([32, 128, 69]) = (B, T, V) — 묶음 32 · 글자 자리 128 · 글자 종류 69
B, T, V = logits.shape

# 점수와 정답을 "문제 B*T 개" 모양으로 편다 (문서 2-2).
# cross_entropy 는 점수 (N, V) · 정답 (N,) 을 기대한다. 줄 · 자리를 구분할 이유가 없으니 앞 두 축을 합친다.
logits_flat = logits.view(B * T, V)
# 결과: logits_flat.shape = torch.Size([4096, 69]) — 문제 4096개 × 보기 69개
targets_flat = yb.view(B * T)
# 결과: targets_flat.shape = torch.Size([4096]) — 문제 4096개의 정답

# 틀린 정도를 숫자 하나로 만든다 (문서 2-1).
# 점수는 확률로 바꾸기 전 그대로 넣는다. 확률 변환 · 정답 글자 확률 꺼내기 · 평균은 함수가 안에서 한다.
loss = F.cross_entropy(logits_flat, targets_flat)
# 결과: loss = 4.3236 (스칼라 텐서) — 학습 전이라 ln(69) ≈ 4.23 근처. 묶음이 무작위라 실행마다 조금 다르다 (문서 2-3)


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
# 결과: loss 4.34 → loss_after 4.09 — 한 걸음만으로 내려감 (값은 묶음이 무작위라 실행마다 조금 다르다)


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
        # 결과: 학습 직후 result = {"train": 0.18.., "val": 0.18..} 꼴 — 아래 train() 출력 참고

    # 점검이 끝났으니 학습 모드로 되돌린다. 안 되돌리면 이후 학습이 dropout 없이 돈다 (문서 4-3 ④).
    model.train()
    return result


# ---- 조각 5: 학습한 모델 저장 ----

# 저장할 파일 경로. checkpoints/ 폴더는 .gitignore 에 있어 리포에 안 올라간다 (문서 5-1 ③).
checkpoint_path = Path("checkpoints/model.pt")


def save_model():
    """모델이 배운 숫자만 파일에 적는다 (문서 5-2 ① ②)."""
    # 폴더가 없으면 만든다. 이미 있어도 에러 없이 넘어간다 (exist_ok=True).
    checkpoint_path.parent.mkdir(exist_ok=True)

    # 모델의 "이름 → 숫자 덩어리" 사전을 뽑는다. 구조는 코드에 있으니 숫자만 있으면 된다 (문서 5-2 ①).
    weights = model.state_dict()
    # 결과: 사전 하나 · 숫자 덩어리 827461개분 (임베딩 · 어텐션 · FFN · LayerNorm · 출력층 가중치)

    # 그 사전을 파일에 적는다. 인자는 (저장할 것, 경로) 순서.
    torch.save(weights, checkpoint_path)
    # 결과: checkpoints/model.pt 생성 · 약 3.2MB (827461개 × 4바이트)


def check_saved_model():
    """저장한 파일을 빈 모델에 불러와 학습한 모델과 같은 점수가 나오는지 본다 (문서 5-2 ③ ④)."""
    # 같은 설정으로 만든 빈 모델. 숫자는 아직 무작위다.
    fresh = GPT(model_config).to(device)

    # 파일의 숫자를 빈 모델에 채워 넣는다. 파일은 torch.load 로 읽는다 (map_location 은 GPU 로 바로 읽으라는 뜻).
    fresh.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # dropout 이 켜져 있으면 점수가 매번 달라지니 둘 다 점검 모드로 바꾸고 같은 입력을 넣는다.
    model.eval()
    fresh.eval()
    with torch.no_grad():
        return torch.equal(model(xb), fresh(xb))
        # 결과: True — 불러온 모델의 점수가 학습한 모델과 한 칸도 다르지 않다


def train():
    """묶음 뽑기 → 점수 → 손실 → 고치기를 max_iters 번 되풀이한다 (문서 3-4)."""
    # 몇 번 되풀이할지는 TrainConfig 의 max_iters (3000) 가 정한다.
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
            # 결과: step 0 → train 3.93 · val 3.94 / 300 → 0.68 · 0.68 / 600 → 0.24 · 0.24 / 900 → 0.19 · 0.19 / 2700 → 0.18 대 · 0.19 대 (묶음이 무작위라 소수 셋째 자리부터 실행마다 다르다)
            # 둘이 거의 같다 — 대화가 학습용 · 검증용에 다 들어 있는 토이 데이터라서 (5단계). 외운 게 아니라는 증거는 못 된다


# 직접 실행했을 때만 확인용 출력 (uv run python -m llm.train)
if __name__ == "__main__":
    print(device)  # 기대: cuda (GPU 를 못 잡으면 cpu — 그러면 CUDA 판 torch 설치를 먼저 본다)
    # 모델 첫 숫자 덩어리가 어느 장치에 있는지 — device 와 같아야 한다
    print(next(model.parameters()).device)  # 기대: cuda:0 (cuda 의 0번 GPU)
    # 모델 안의 파라미터 총 개수 — 4단계에서 센 값과 같아야 한다
    print(sum(p.numel() for p in model.parameters()))  # 기대: 827461
    # 옵티마이저가 들고 있는 학습률 — TrainConfig 의 값과 같아야 한다
    print(optimizer.param_groups[0]["lr"])  # 기대: 0.0003
    # 틀린 정도 — 학습 전이라 69글자를 찍어 맞히는 값 ln(69) 근처여야 한다
    print(loss.item())  # 기대: 약 4.23 (4.1 ~ 4.6 — 묶음이 무작위라 실행마다 조금 다르다)
    # 펴기 전 · 후 모양 — (B, T, V) 가 (B*T, V) 로
    print(logits.shape, logits_flat.shape, targets_flat.shape)  # 기대: [32,128,69] [4096,69] [4096]
    # 한 번 고친 뒤 같은 묶음으로 다시 잰 손실 — 위 loss 보다 작아야 한다
    print(loss_after.item())  # 기대: loss 보다 작은 값 (한 걸음만 갔으니 조금만)
    # 반복 학습 — 손실이 4 대에서 내려가는지 본다
    train()  # 기대: step 0 ~ 2700 까지 300 간격으로 10줄 · 값이 점점 작아짐
    # 학습이 끝난 숫자를 파일로 적고, 불러온 모델과 점수를 비교한다
    save_model()
    print(check_saved_model())  # 기대: True (같은 점수)
