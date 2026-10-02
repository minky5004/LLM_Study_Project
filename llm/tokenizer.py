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


# ---- 조각 3: encode · decode (표 두 장으로 텍스트 <-> 번호 목록) ----

def encode(s):
    """문자열 -> 번호 리스트.  예) "안녕" -> [40, 15]

    shape: 길이 L 인 문자열이 길이 L 인 리스트가 된다 (글자 하나 = 번호 하나라 길이가 그대로).
    나중에 이 리스트를 torch.tensor 로 감싸면 모델 입력(shape: (L,))이 된다.
    """
    # 글자를 하나씩 꺼내(for c in s) stoi 표에서 번호를 찾아 리스트로 모은다.
    # 리스트 컴프리헨션 [식 for 변수 in 반복할것] — 딕셔너리판({키: 값 for ...})과 같은 틀이다
    return [stoi[c] for c in s]


def decode(ids):
    """번호 리스트 -> 문자열.  예) [40, 15] -> "안녕"  (encode 의 정반대)"""
    # 번호를 하나씩 꺼내(for i in ids) itos 표에서 글자를 찾는다. 여기까지는 글자 리스트(['안', '녕'])
    chars_back = [itos[i] for i in ids]
    # 글자 리스트를 구분자 없이 이어 붙여 문자열 하나로 만든다.
    # "구분자".join(리스트) 꼴 — 구분자 자리에 빈 문자열을 넣으면 사이에 아무것도 안 끼고 붙는다
    return "".join(chars_back)


# 갔다 오면 제자리여야 한다 — decode(encode(s)) == s 가 토크나이저가 맞는지 보는 가장 간단한 시험
if __name__ == "__main__":
    print(encode("안녕"))                    # 기대: [40, 15]
    print(decode(encode("안녕")))            # 기대: 안녕
    print(decode(encode(text)) == text)      # 기대: True (train.txt 65572자 전체도 왕복)
