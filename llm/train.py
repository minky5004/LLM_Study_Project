"""학습 루프.

지금은 조각 1까지: 장치 · 모델 · 옵티마이저를 준비한다 (읽을 것은 6단계 문서의 1-1 ~ 1-3).
틀린 정도 계산 · 숫자 고치기 · 반복은 이어지는 조각에서 이 파일에 붙인다.
"""

import torch

from llm.config import ModelConfig, TrainConfig
from llm.model import GPT

# 주의: tokenizer 를 import 하면 그 모듈 최상위가 실행되어 train.txt 를 읽는다. 프로젝트 루트에서 `-m` 으로 실행해야 경로가 맞는다.
# vocab_size: 토크나이저가 글에서 센 글자 종류 수. 5단계에서 한 것처럼 모델 설정에 넘겨 config.py 의 손으로 적은 69 에 기대지 않는다.
from llm.tokenizer import vocab_size


# ---- 조각 1: 장치 · 모델 · 옵티마이저 준비 ----

# 계산을 할 장치를 고른다 (문서 1-1).
# GPU(NVIDIA) 를 쓸 수 있으면 "cuda" · 없으면 "cpu". 이 PC 는 RTX 3060 이라 "cuda" 가 나와야 한다.
# [빈칸 1] torch.cuda 에서 "GPU 를 쓸 수 있는가" 를 True / False 로 알려 주는 함수 이름을 넣는다. (힌트: is_ 로 시작)
device = "cuda" if torch.cuda.____() else "cpu"

# 설정 상자 두 개를 만든다. 모델 크기는 ModelConfig · 학습 방법은 TrainConfig 가 들고 있다.
# 글자 종류 수는 토크나이저가 센 값으로 덮어쓴다 (5단계 4-1 과 같다).
model_config = ModelConfig(vocab_size=vocab_size)
train_config = TrainConfig()

# 모델을 만들어 장치로 옮긴다 (문서 1-2).
# 모델의 숫자(파라미터)가 GPU 메모리에 있어야 GPU 가 계산할 수 있다. 한 번 옮기면 끝이다.
# [빈칸 2] 모델을 장치로 옮기는 함수 이름을 넣는다. device 를 인자로 받는다. (힌트: "~로" 라는 뜻의 영어 전치사)
model = GPT(model_config).____(device)

# 옵티마이저를 만든다 (문서 1-3).
# AdamW 에 "고칠 숫자 목록" 과 "한 번에 얼마나 고칠지(학습률)" 를 넘긴다. 아직 고치지는 않는다 — 고치는 일은 조각 3 에서 한다.
# [빈칸 3] 모델 안의 고칠 수 있는 숫자 전부를 돌려주는 함수를 호출한다. (힌트: parameter 를 복수형으로 · 괄호 필요)
# lr 에는 TrainConfig 에 든 학습률을 꺼내 넘긴다 (3e-4).
optimizer = torch.optim.AdamW(model.____, lr=train_config.learning_rate)


# 직접 실행했을 때만 확인용 출력 (uv run python -m llm.train)
if __name__ == "__main__":
    print(device)  # 기대: cuda (GPU 를 못 잡으면 cpu — 그러면 CUDA 판 torch 설치를 먼저 본다)
    # 모델 첫 숫자 덩어리가 어느 장치에 있는지 — device 와 같아야 한다
    print(next(model.parameters()).device)  # 기대: cuda:0 (cuda 의 0번 GPU)
    # 모델 안의 파라미터 총 개수 — 4단계에서 센 값과 같아야 한다
    print(sum(p.numel() for p in model.parameters()))  # 기대: 827461
    # 옵티마이저가 들고 있는 학습률 — TrainConfig 의 값과 같아야 한다
    print(optimizer.param_groups[0]["lr"])  # 기대: 0.0003
