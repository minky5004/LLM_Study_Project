"""텍스트 <-> 토큰(정수) 변환.

모델은 글자를 못 읽고 숫자만 읽는다. 그래서 글자마다 고유 번호(토큰)를 붙여
텍스트 -> 번호 목록(encode) · 번호 목록 -> 텍스트(decode) 로 오가게 해 주는 번역기가 필요하다.
이 프로젝트는 글자 하나 = 토큰 하나인 "글자 단위" 토크나이저부터 만든다.
"""

# 2단계에서 만든 학습 텍스트 경로. 글자 종류를 세는 데 쓴다
TRAIN_PATH = "data/train.txt"


# ---- 조각 1: 글자 집합(vocab) 만들기 ----

# 파일을 통째로 문자열 하나로 읽는다 (65572자 · shape 개념 없음 — 그냥 긴 str).
# encoding="utf-8" 을 적는 이유: 안 적으면 윈도우 기본(cp949)으로 읽어 한글이 깨진다 (2단계에서 배운 것)
text = open(TRAIN_PATH, encoding="utf-8").read()

# 글자 종류를 중복 없이 모으고, 번호를 매길 순서를 고정한다.
#   text          : "사용자: 배고파\n봇: ..." — 글자가 65572개 (같은 글자 여러 번 등장)
#   set(text)     : 글자를 하나씩 꺼내 중복 없이 모은 집합 — 순서는 보장되지 않는다
#   sorted(집합)  : 그 집합을 정렬된 리스트로 바꾼다 — 실행마다 같은 순서가 나오게
# 번호는 "리스트에서 몇 번째냐" 로 정해지므로, 순서가 흔들리면 어제 학습한 모델과 오늘의 토크나이저가 어긋난다
chars = sorted(set(text))

# 글자 종류의 개수 = vocab_size. ModelConfig.vocab_size(지금 임시값 256)를 이 값으로 바꾸게 된다.
vocab_size = len(chars)

# 직접 실행했을 때만 확인용 출력 (다른 파일이 import 할 때는 실행되지 않는다)
if __name__ == "__main__":
    print(vocab_size)                # 기대: 69
    print(repr("".join(chars[:12]))) # 기대: 줄바꿈 · 공백 · ':' 가 맨 앞 (개행도 엄연한 토큰)


# ---- 조각 2: stoi · itos 사전 (글자 <-> 번호 표 두 장) ----

# chars 는 리스트라 "몇 번째 글자냐" 는 알지만, 글자로 번호를 찾으려면 처음부터 훑어야 한다.
# 딕셔너리로 만들어 두면 글자를 넣자마자 번호가 바로 나온다.
#
# stoi (string to int): 글자 -> 번호.  예) {'\n': 0, ' ': 1, ':': 2, '가': 3, ...}  · encode 가 쓴다
#   enumerate(chars) 는 chars 의 글자마다 (번호, 글자) 쌍을 0번부터 붙여서 하나씩 꺼내준다.
#   예) enumerate(['녕', '세']) -> (0, '녕'), (1, '세')
#   {키: 값 for 변수 in 반복할것} 은 for 로 돌며 딕셔너리를 한 줄로 만드는 문법이다.
#   for 뒤 변수 i, ch 는 enumerate 쌍의 순서 (번호, 글자) 그대로 받는다. 키 = 글자(ch) · 값 = 번호(i)
stoi = {ch: i for i, ch in enumerate(chars)}

# itos (int to string): 번호 -> 글자.  예) {0: '\n', 1: ' ', 2: ':', 3: '가', ...}  · decode 가 쓴다
#   stoi 와 같은 enumerate 쌍에서 키와 값 자리만 맞바꾸면 된다.
itos = {i: ch for i, ch in enumerate(chars)}

# 두 표는 서로의 거울이다 — 표 크기는 vocab_size 와 같아야 하고,
# 어떤 글자든 itos[stoi[글자]] 로 갔다 오면 제자리로 돌아와야 한다.
if __name__ == "__main__":
    print(len(stoi), len(itos))                # 기대: 69 69 (vocab_size 와 같다)
    print(stoi["\n"], stoi[" "], stoi[":"])    # 기대: 0 1 2
    print(itos[stoi["가"]])                    # 기대: 가 (갔다 오면 제자리)
