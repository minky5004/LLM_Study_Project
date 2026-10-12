"""학습 데이터 로딩 · 배치 구성.

지금은 조각 1까지: 글 전체를 번호 텐서로 바꾸고 학습용 · 검증용으로 나눈다 (읽을 것은 5단계 문서의 1-1 ~ 1-3).
조각 2 부터 입력 x · 정답 y 를 만들고, 조각 3 에서 묶음(배치)으로 뽑는 함수를 붙인다.
"""

import torch

# text: tokenizer.py 가 train.txt 를 읽어 둔 글 전체 (65572자).
# encode: 글 -> 번호 목록(파이썬 리스트) · decode: 번호 목록 -> 글.
# 주의: tokenizer 를 import 하면 그 모듈 최상위가 실행되어 train.txt 를 읽는다. 프로젝트 루트에서 `-m` 으로 실행해야 경로가 맞는다.
from llm.tokenizer import decode, encode, text


# ---- 조각 1: 글 전체를 번호 텐서로 + 학습용 · 검증용으로 나누기 ----

# 글 전체를 번호 목록으로 바꾼 뒤 텐서로 감싼다 (문서 1-1).
# 번호는 "임베딩 표의 몇 번째 줄" 이라 정수여야 한다. nn.Embedding 은 정수만 받는다.
# dtype 에 64비트 정수 타입 torch.long 을 지정한다 (소수 타입이면 임베딩 표에서 에러가 난다).
data = torch.tensor(encode(text), dtype=torch.long)
# 결과: data.shape = torch.Size([65572]) · data.dtype = torch.int64 (torch.long 과 같은 타입)
#       data[:13] = [35, 50, 57, 2, 1, 31, 5, 65, 0, 32, 2, 1, 27] — "사용자: 배고파\n봇: 물"

# 앞 90% 는 학습용 · 뒤 10% 는 검증용으로 가를 "자리 번호" 를 구한다 (문서 1-2 · 1-3).
# 0.9 * len(data) 는 소수(59014.8)가 나오는데 자리 번호는 정수여야 해서 int() 로 소수점 아래를 버린다.
n = int(0.9 * len(data))
# 결과: n = 59014 (0.9 * 65572 = 59014.8 에서 소수점 아래를 버림)

# 슬라이스 [시작:끝] — 시작 자리부터 끝 자리 "앞까지". 앞이나 뒤를 비우면 처음부터 · 끝까지다 (문서 1-3).
# 학습용: 처음부터 n 앞까지 · 검증용: n 부터 끝까지. 두 구간은 겹치지 않고 합치면 전체와 같다.
train_data = data[:n]
val_data = data[n:]
# 결과: len(train_data) = 59014 · len(val_data) = 6558 (합 65572 — 이어 붙이면 원본과 torch.equal 로 같다)
#       val_data 맨 앞 20자 = ' 이름은 아직 없어요\n\n사용자: 너는' — 90% 지점이 대화 경계와 무관해서 문장 중간에서 시작한다
#       이 토이 데이터는 서로 다른 대화 46종류를 3000번 반복한 것이라, 검증용 대화 296개가 전부 학습용에도 똑같이 있다
#       (나누는 구조를 익히는 데 의미가 있고 "외웠나 배웠나" 를 엄밀히 가르지는 못한다)


# 직접 실행했을 때만 확인용 출력 (uv run python -m llm.dataset)
if __name__ == "__main__":
    print(data.shape, data.dtype)  # 기대: torch.Size([65572]) torch.int64
    print(len(train_data), len(val_data))  # 기대: 59014 6558
    print(len(train_data) + len(val_data) == len(data))  # 기대: True (겹침도 빠짐도 없다)
    # 텐서를 파이썬 리스트로 되돌려 decode 에 넣는다. 검증용 글의 맨 앞이 어떤 글인지 눈으로 본다.
    print(repr(decode(val_data[:20].tolist())))  # 기대: 사람이 읽을 수 있는 대화 글 20자

# ---- 직접 실행 결과 (uv run python -m llm.dataset) ----
# torch.Size([65572]) torch.int64
# 59014 6558
# True
# ' 이름은 아직 없어요\n\n사용자: 너는'
