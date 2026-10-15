"""학습 루프.

지금은 조각 2까지: 장치 · 모델 · 옵티마이저 준비 (문서 1-1 ~ 1-3) · 틀린 정도 계산 (문서 2-1 ~ 2-3).
숫자 고치기 · 반복은 이어지는 조각에서 이 파일에 붙인다.
"""

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
    print(loss.item())  # 기대: 약 4.23 (4.1 ~ 4.4)
    # 펴기 전 · 후 모양 — (B, T, V) 가 (B*T, V) 로
    print(logits.shape, logits_flat.shape, targets_flat.shape)  # 기대: [32,128,69] [4096,69] [4096]
