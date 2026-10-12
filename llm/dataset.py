"""학습 데이터 로딩 · 배치 구성.

지금은 조각 2까지: 글 전체를 번호 텐서로 바꾸고 학습용 · 검증용으로 나눈다 (읽을 것은 5단계 문서의 1-1 ~ 1-3).
그다음 입력 x · 정답 y 를 한 칸 밀어 잘라 낸다 (문서 2-1 ~ 2-4). 조각 3 에서 묶음(배치)으로 뽑는 함수를 붙인다.
"""

import torch

from llm.config import ModelConfig

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


# ---- 조각 2: 입력 x 와 정답 y — 한 칸 민 쌍 ----

# 한 번에 자를 글자 수 = 모델이 한 번에 볼 수 있는 글자 수의 최댓값 (config 의 block_size = 128).
# 모델 크기를 바꾸면 여기도 따라 바뀌도록 숫자를 직접 적지 않고 config 에서 꺼낸다.
block_size = ModelConfig().block_size

# 학습용 글에서 잘라 올 구간의 "시작 위치". 지금은 맨 앞(0)에서 한 번 잘라 보고, 조각 3 에서 무작위로 바꾼다.
i = 0

# 입력 x: 시작 위치 i 부터 block_size 글자. 정답 y: x 와 같은 길이를 "한 칸 뒤로 민" 구간 (문서 2-2).
# 슬라이스 [시작:끝] 은 끝 "앞까지" 라서 길이가 (끝 - 시작) 이다. 둘 다 길이가 block_size 가 되게 시작과 끝을 적는다.
# 정답 y 의 자리 하나는 x 의 같은 자리 글자의 "바로 다음 글자" 다 (문서 2-2 · 2-3).
x = train_data[i : i + block_size]
y = train_data[i + 1 : i + block_size + 1]
# 결과: x.shape = y.shape = torch.Size([128])
#       decode(x[:13]) = '사용자: 배고파\n봇: 물' · decode(y[:13]) = '용자: 배고파\n봇: 물풀' — y 는 같은 자리 x 글자의 바로 다음 글자
#       torch.equal(y[:-1], x[1:]) = True (y 의 앞 127개 = x 의 뒤 127개 — 한 칸 밀렸다는 증거)

# 시작 위치로 쓸 수 있는 자리의 개수 (문서 2-4). y 는 x 보다 한 글자 더 뒤까지 필요하다.
# 시작 위치 i 에서 y 의 끝은 i + block_size + 1 (끝 앞까지) 이라, 이게 글 길이를 넘으면 y 가 모자란다.
# 그러면 i 는 0 ~ (글 길이 - block_size - 1) 이고, 0 부터 세므로 이 범위의 개수는 (글 길이 - block_size) 다 (-1 과 +1 이 서로 없어진다).
n_starts = len(train_data) - block_size
# 결과: n_starts = 58886 (= 59014 - 128)
#       경계 확인: 마지막 시작 위치(58885)에서 y 는 정확히 128개 · 한 칸만 더 가면 127개로 모자란다


# 직접 실행했을 때만 확인용 출력 (uv run python -m llm.dataset)
if __name__ == "__main__":
    print(data.shape, data.dtype)  # 기대: torch.Size([65572]) torch.int64
    print(len(train_data), len(val_data))  # 기대: 59014 6558
    print(len(train_data) + len(val_data) == len(data))  # 기대: True (겹침도 빠짐도 없다)
    # 텐서를 파이썬 리스트로 되돌려 decode 에 넣는다. 검증용 글의 맨 앞이 어떤 글인지 눈으로 본다.
    print(repr(decode(val_data[:20].tolist())))  # 기대: 사람이 읽을 수 있는 대화 글 20자

    # 조각 2 확인: x 와 y 는 길이가 같고, y 는 x 를 한 칸 민 것이다
    print(x.shape, y.shape)  # 기대: torch.Size([128]) torch.Size([128])
    print(repr(decode(x[:13].tolist())))  # 기대: '사용자: 배고파\n봇: 물' (글 맨 앞 13자)
    print(repr(decode(y[:13].tolist())))  # 기대: '용자: 배고파\n봇: 물풀' (같은 자리 글자의 바로 다음 글자)
    print(torch.equal(y[:-1], x[1:]))  # 기대: True (y 의 앞 127개 = x 의 뒤 127개)
    print(n_starts)  # 기대: 58886 (= 59014 - 128)
    # 경계: 마지막 시작 위치에서는 y 가 정확히 128개 · 한 칸만 더 가면 127개로 모자란다
    last = n_starts - 1
    print(len(train_data[last + 1 : last + block_size + 1]))  # 기대: 128
    print(len(train_data[last + 2 : last + block_size + 2]))  # 기대: 127

# ---- 직접 실행 결과 (uv run python -m llm.dataset) ----
# torch.Size([65572]) torch.int64
# 59014 6558
# True
# ' 이름은 아직 없어요\n\n사용자: 너는'
# torch.Size([128]) torch.Size([128])
# '사용자: 배고파\n봇: 물'
# '용자: 배고파\n봇: 물풀'
# True
# 58886
# 128
# 127
