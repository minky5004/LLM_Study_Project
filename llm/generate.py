"""추론·텍스트 생성.

지금은 조각 2까지: 저장한 모델 불러오기 · 시작 글을 번호 텐서로 (문서 1-1 ~ 1-2) · 다음 글자 하나 뽑기 (문서 2-1 ~ 2-2).
이어 쓰기 · 대화는 이어지는 조각에서 이 파일에 붙인다.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from llm.config import ModelConfig
from llm.model import GPT

# 주의: tokenizer 를 import 하면 그 모듈 최상위가 실행되어 train.txt 를 읽는다. 프로젝트 루트에서 `-m` 으로 실행해야 경로가 맞는다.
from llm.tokenizer import decode, encode, vocab_size

# 6단계 train.py 를 import 하지 않는다 — import 만 해도 모델을 만들고 한 걸음 학습하는 줄이 최상위에 있어서다.
# 필요한 것(장치 · 설정 · 파일 경로)은 여기서 다시 정한다.


# ---- 조각 1: 저장한 모델 불러오기 · 시작 글을 번호 텐서로 ----

# 학습 때와 같은 장치를 고른다 (6단계 문서 1-1).
device = "cuda" if torch.cuda.is_available() else "cpu"

# 학습 때와 똑같은 설정으로 빈 모델을 만든다. 구조가 다르면 저장한 숫자가 들어갈 칸이 안 맞아 에러가 난다 (6단계 문서 5-2 ③).
model_config = ModelConfig(vocab_size=vocab_size)
model = GPT(model_config).to(device)

# 6단계에서 저장한 파일. 학습한 숫자 827461개가 들어 있다.
checkpoint_path = Path("checkpoints/model.pt")

# 파일의 숫자를 빈 모델에 채워 넣는다 (문서 1-1 ②). 파일은 torch.load 로 읽는다.
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# 결과: <All keys matched successfully> — 저장한 숫자 827461개가 빈 모델 칸에 그대로 들어감

# 글을 만들 때는 dropout 을 꺼야 한다 — 모델을 점검 모드로 바꾼다 (6단계 문서 4-3 ③).
model.eval()

# 글을 이어 쓰게 할 시작 글. 학습 데이터와 같은 모양("사용자: ...\n봇:")이라 모델이 봇의 답을 이어 쓰게 된다.
prompt = "사용자: 배고파\n봇:"

# 시작 글을 모델이 받는 모양 (1, T) 의 번호 텐서로 만든다 (문서 1-2).
# encode(prompt) 는 길이 T 인 번호 리스트 하나를 돌려준다. 모델은 (B, T) — "문장 B개 묶음" 을 받는다.
# 문장이 한 개뿐이니 B = 1 이 되도록 리스트를 한 겹 더 감싼다. dtype 은 번호이니 long, device 는 모델과 같은 곳.
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
# 결과: idx.shape = torch.Size([1, 11]) — 문장 1개 · 글자 11개 (시작 글 "사용자: 배고파" + 줄바꿈 + "봇:")

# 점수만 재는 일이라 기록(no_grad)을 끈다 (6단계 문서 4-3 ②). 모델에 넣으면 글자 자리마다 69개 점수가 나온다.
with torch.no_grad():
    logits = model(idx)
# 결과: logits.shape = torch.Size([1, 11, 69]) — 글자 자리 11개마다 69개 점수


# ---- 조각 2: 다음 글자 하나 뽑기 ----

# 점수 (1, 11, 69) 에서 맨 마지막 글자 자리의 69개 점수만 꺼낸다 (문서 2-1).
# 시작 글을 끝까지 읽은 뒤에 올 글자가 궁금하니 마지막 자리다. 묶음 칸은 그대로 · 글자 자리는 맨 끝 · 점수는 전부.
last = logits[:, -1, :]
# 결과: last.shape = torch.Size([1, 69]) — 마지막 자리의 점수 69개

# 69개 점수를 확률로 바꾼다 — 0 이상 · 합이 1 (문서 2-2 ②). 69개가 늘어선 방향은 맨 끝 칸이다.
probs = F.softmax(last, dim=-1)
# 결과: probs.shape = torch.Size([1, 69]) · probs.sum() = 1.0000001 (소수 계산 오차) · 공백 확률 0.9999

# 확률대로 글자 번호 하나를 뽑는다 (문서 2-2 ③). 확률 0.7 인 글자는 10번 중 약 7번 뽑힌다.
next_idx = torch.multinomial(probs, num_samples=1)
# 결과: next_idx.shape = torch.Size([1, 1]) · decode 하면 ' ' (공백) — "봇:" 뒤는 학습 데이터 3000개가 전부 공백이라 3번 실행해도 같다


# 직접 실행했을 때만 확인용 출력 (uv run python -m llm.generate)
if __name__ == "__main__":
    print(idx.shape)  # 기대: torch.Size([1, 11]) — 문장 1개 · 글자 11개 ("사용자: 배고파\n봇:")
    print(decode(idx[0].tolist()))  # 기대: 시작 글이 그대로 — 번호로 바꿨다 되돌려 제자리인지 확인
    print(logits.shape)  # 기대: torch.Size([1, 11, 69]) — 글자 자리마다 69개 점수
    print(last.shape)  # 기대: torch.Size([1, 69]) — 마지막 자리의 점수 69개
    print(probs.sum().item())  # 기대: 1.0 근처 — 확률은 합이 1
    print(next_idx.shape)  # 기대: torch.Size([1, 1]) — 뽑힌 글자 번호 1개
    print(repr(decode(next_idx[0].tolist())))  # 기대: ' ' (공백) — "봇:" 뒤는 공백이 거의 확실
