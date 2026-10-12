"""학습 데이터 로딩 · 배치 구성.

지금은 조각 3까지: 글 전체를 번호 텐서로 바꾸고 학습용 · 검증용으로 나눈다 (읽을 것은 5단계 문서의 1-1 ~ 1-3).
그다음 입력 x · 정답 y 를 한 칸 밀어 잘라 낸다 (문서 2-1 ~ 2-4).
마지막으로 무작위 위치 여러 곳에서 잘라 묶음(배치)으로 쌓아 주는 get_batch 함수를 만든다 (문서 3-1 ~ 3-4).
"""

import torch

from llm.config import ModelConfig, TrainConfig

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


# ---- 조각 3: get_batch — 무작위 위치 여러 곳에서 잘라 쌓기 ----

# 한 번에 몇 쌍을 묶어 풀지 (config 의 batch_size = 32). block_size 와 마찬가지로 config 에서 꺼낸다.
batch_size = TrainConfig().batch_size


def get_batch(split):
    """학습용 또는 검증용 글에서 무작위 위치 batch_size 곳을 잘라 (x, y) 묶음을 돌려준다.

    split: "train" 이면 학습용 글 · "val" 이면 검증용 글에서 뽑는다.
    반환: x, y 모두 shape (batch_size, block_size) = (32, 128) · dtype int64.
          y 는 x 를 한 칸 민 정답이다 (조각 2).
    """
    # 어느 글에서 뽑을지. 둘 외의 글자가 들어오면 val 로 조용히 넘어가지 않게 바로 멈춘다.
    assert split in ("train", "val"), 'split 은 "train" 또는 "val" 이어야 한다'
    source = train_data if split == "train" else val_data

    # 시작 위치를 batch_size 개 무작위로 뽑는다 (문서 3-2).
    # torch.randint(상한, (개수,)) 는 0 이상 "상한 미만" 의 정수를 개수만큼 뽑는다. 상한 자체는 안 나온다.
    # 조각 2 에서 구한 "y 가 모자라지 않는 시작 위치의 개수" 를 상한으로 쓴다.
    # 검증용 글은 학습용보다 짧아서 개수가 달라지므로, 숫자를 고정하지 않고 뽑는 글(source)의 길이로 계산한다.
    ix = torch.randint(len(source) - block_size, (batch_size,))
    # 결과: ix.shape = torch.Size([32]) · 값은 0 이상 (len(source) - 128) 미만 — 학습용이면 0 ~ 58885 · 검증용이면 0 ~ 6429

    # 뽑은 위치마다 x 를 잘라 위아래로 쌓는다 (문서 3-3).
    # [source[i : i + block_size] for i in ix] 는 위치마다 길이 block_size 짜리 조각을 만든 리스트 (컴프리헨션).
    # torch.stack 은 같은 길이의 조각들을 "새 축을 만들어" 쌓는다. (128,) 짜리 32개 -> (32, 128)
    x = torch.stack([source[i : i + block_size] for i in ix])
    # 결과: x.shape = torch.Size([32, 128]) · dtype int64 — 줄 하나하나는 글의 서로 다른 곳에서 따 온 128글자

    # 같은 위치 ix 로 y 도 자른다 (문서 3-4). 위치를 다시 뽑으면 x 와 y 가 어긋나 한 칸 민 쌍이 아니게 된다.
    y = torch.stack([source[i + 1: i + block_size + 1] for i in ix])
    # 결과: y.shape = torch.Size([32, 128]) · torch.equal(y[:, :-1], x[:, 1:]) = True — 32줄 전부 한 칸 민 쌍
    #       "train" · "val" 각각 300번 뽑아도 모양과 한 칸 밀림이 그대로 · split 이 둘 외의 값이면 AssertionError 로 멈춘다

    # GPU 로 옮기는 일(.to(device))은 6단계 학습 루프에서 한다. 여기서는 CPU 텐서 그대로 돌려준다.
    return x, y


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

    # 조각 3 확인: 묶음 하나
    xb, yb = get_batch("train")
    print(xb.shape, yb.shape)  # 기대: torch.Size([32, 128]) torch.Size([32, 128])
    print(xb.dtype)  # 기대: torch.int64 (임베딩 표에 넣을 수 있는 정수)
    print(torch.equal(yb[:, :-1], xb[:, 1:]))  # 기대: True (32쌍 전부 한 칸 밀린 쌍)
    xv, yv = get_batch("val")
    print(xv.shape, yv.shape)  # 기대: torch.Size([32, 128]) torch.Size([32, 128]) (검증용 글에서도 같은 모양)
    print(torch.equal(get_batch("train")[0], get_batch("train")[0]))  # 기대: False (뽑을 때마다 위치가 다르다)

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
# torch.Size([32, 128]) torch.Size([32, 128])
# torch.int64
# True
# torch.Size([32, 128]) torch.Size([32, 128])
# False
