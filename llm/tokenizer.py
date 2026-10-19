"""텍스트 <-> 토큰(정수) 변환.

모델은 글자를 못 읽고 숫자만 읽는다. 그래서 글자마다 고유 번호(토큰)를 붙여
텍스트 -> 번호 목록(encode) · 번호 목록 -> 텍스트(decode) 로 오가게 해 주는 번역기가 필요하다.
이 프로젝트는 글자 하나 = 토큰 하나인 "글자 단위" 토크나이저부터 만든다.

8단계 조각 2: 학습용 · 검증용 글이 두 파일로 나뉘어(data/make_dataset.py), 둘 다 읽고 글자 목록은 둘을 합쳐 센다 (8단계 문서 2-3).
"""

# 학습용 · 검증용 글의 경로. data/make_dataset.py 가 만든다 (리포에는 없음 — clone 뒤 그 스크립트를 먼저 돌린다).
TRAIN_PATH = "data/train.txt"
VAL_PATH = "data/val.txt"


# ---- 조각 1: 글자 집합(vocab) 만들기 ----

# 파일을 통째로 문자열 하나로 읽는다 (긴 str 하나 — shape 개념 없음).
# 인코딩(글자를 바이트로 저장하는 약속)을 안 적으면 윈도우 기본값(cp949)으로 읽어 한글이 깨진다.
# 두 파일은 utf-8 로 저장했으니 읽을 때도 같은 약속을 적어야 한다.
train_text = open(TRAIN_PATH, encoding="utf-8").read()
val_text = open(VAL_PATH, encoding="utf-8").read()
# 결과: train_text 413,139자 · val_text 46,705자 (2단계 토이 데이터는 train.txt 하나 65,572자)

# 글자 종류를 중복 없이 모으고, 번호를 매길 순서를 고정한다.
#   set(글)     : 글자를 하나씩 꺼내 중복 없이 모은 집합 — 순서는 보장되지 않는다
#   sorted(...) : 그 모음을 정렬된 리스트로 바꾼다 — 실행마다 같은 순서가 나오게
# 번호는 "리스트에서 몇 번째냐" 로 정해지므로, 순서가 흔들리면 어제 학습한 모델과 오늘의 토크나이저가 어긋난다.
# 검증용에만 나오는 글자가 있다. 학습용 글만으로 세면 그 글자는 번호가 없어 검증용을 encode 할 때 KeyError 가 난다 (8단계 문서 2-3).
# 두 글을 + 로 이어 붙인 글에서 글자 종류를 센다.
chars = sorted(set(train_text + val_text))
# 결과: 1244개짜리 리스트 — 맨 앞 ['\n', ' ', '!', '%', "'", ',', '-', '.', '0', ...] · 맨 뒤 [..., '힐', '힘', '힙']
#       (줄바꿈 · 공백 · 기호 · 숫자 · 영어가 먼저, 한글은 가나다순)
#       학습용 글만으로 셌다면 검증용을 encode 할 때 KeyError: '휨' 으로 멈춘다 (실행 확인)

# 글자 종류의 개수 = vocab_size. 모델을 만들 때 이 값을 넘긴다 (config.py 의 69 는 손으로 적은 기본값 — 결정 기록 0007).
vocab_size = len(chars)
# 결과: 1244 (2단계 토이 데이터는 69)

# 직접 실행했을 때만 확인용 출력 (다른 파일이 import 할 때는 실행되지 않는다)
if __name__ == "__main__":
    print(vocab_size)                 # 결과: 1244
    print(repr("".join(chars[:12])))  # 맨 앞 12글자를 이어 붙여 본다 (repr = 줄바꿈도 \n 으로 보이게)
    # 결과: "\n !%',-.0123" — 줄바꿈 · 공백 · 기호 · 숫자 순
    # 검증용에만 있는 글자 — 학습용만으로 셌다면 이 글자들에서 KeyError 가 났다
    print(len(set(val_text) - set(train_text)))  # 결과: 22


# ---- 조각 2: stoi · itos 사전 (글자 <-> 번호 표 두 장) ----

# chars 는 리스트라 "몇 번째 글자냐" 는 알지만, 글자로 번호를 찾으려면 처음부터 훑어야 한다.
# 딕셔너리(키로 값을 바로 찾는 표)로 만들어 두면 글자를 넣자마자 번호가 바로 나온다.
#
# stoi (string to int): 글자 -> 번호.  예) {'\n': 0, ' ': 1, ...}  · encode 가 쓴다
#   enumerate(chars) 는 chars 의 글자마다 (번호, 글자) 쌍을 0번부터 붙여서 하나씩 꺼내준다.
#   {키: 값 for 변수 in 반복할것} 은 for 로 돌며 딕셔너리를 한 줄로 만드는 문법이다.
#   for 뒤 i, ch 는 쌍의 순서 (번호, 글자) 그대로 받는다. 키 = 글자(ch) · 값 = 번호(i)
stoi = {ch: i for i, ch in enumerate(chars)}
# 결과: {'\n': 0, ' ': 1, '!': 2, ... '힙': 1243} — 글자 1244개가 0~1243번

# itos (int to string): 번호 -> 글자.  예) {0: '\n', 1: ' ', ...}  · decode 가 쓴다
#   stoi 와 같은 쌍에서 키와 값 자리만 맞바꾸면 된다.
itos = {i: ch for i, ch in enumerate(chars)}
# 결과: {0: '\n', 1: ' ', 2: '!', ... 1243: '힙'} — stoi 의 키와 값을 맞바꾼 표

# 두 표는 서로의 거울이다 — 표 크기는 vocab_size 와 같아야 하고,
# 어떤 글자든 itos[stoi[글자]] 로 갔다 오면 제자리로 돌아와야 한다.
if __name__ == "__main__":
    print(len(stoi), len(itos))               # 결과: 1244 1244 (vocab_size 와 같다)
    print(stoi["\n"], stoi[" "])              # 결과: 0 1 (줄바꿈 · 공백이 맨 앞 번호)
    print(itos[stoi["가"]])                   # 결과: 가 (갔다 오면 제자리)


# ---- 조각 3: encode · decode (표 두 장으로 텍스트 <-> 번호 목록) ----

def encode(s):
    """문자열 -> 번호 리스트.  예) "안녕" -> [752, 241]

    shape: 길이 L 인 문자열이 길이 L 인 리스트가 된다 (글자 하나 = 번호 하나라 길이가 그대로).
    나중에 이 리스트를 torch.tensor 로 감싸면 모델 입력(shape: (L,))이 된다.
    """
    # 글자를 하나씩 꺼내(for c in s) 번호로 바꿔 리스트로 모은다.
    # 리스트 컴프리헨션 [식 for 변수 in 반복할것] — 딕셔너리판({키: 값 for ...})과 같은 틀이다
    return [stoi[c] for c in s]


def decode(ids):
    """번호 리스트 -> 문자열.  예) [752, 241] -> "안녕"  (encode 의 정반대)"""
    # 번호를 하나씩 꺼내(for i in ids) 글자로 바꾼다. 여기까지는 글자 리스트(['안', '녕'])
    chars_back = [itos[i] for i in ids]
    # 글자 리스트를 구분자 없이 이어 붙여 문자열 하나로 만든다.
    # "구분자".join(리스트) 꼴 — 구분자 자리에 빈 문자열을 넣으면 사이에 아무것도 안 끼고 붙는다
    return "".join(chars_back)


# 갔다 오면 제자리여야 한다 — decode(encode(s)) == s 가 토크나이저가 맞는지 보는 가장 간단한 시험
if __name__ == "__main__":
    print(encode("안녕"))                              # 결과: [752, 241]
    print(decode(encode("안녕")))                      # 결과: 안녕
    print(decode(encode(train_text)) == train_text)    # 결과: True (학습용 글 413,139자 전체도 왕복)
    print(decode(encode(val_text)) == val_text)        # 결과: True (검증용 글도 — 검증용에만 있는 글자까지 번호가 있다)

# ---- 직접 실행 결과 (uv run python -m llm.tokenizer) ----
# 1244
# "\n !%',-.0123"
# 22
# 1244 1244
# 0 1
# 가
# [752, 241]
# 안녕
# True
# True
