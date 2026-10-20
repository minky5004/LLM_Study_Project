"""BPE 토크나이저 (9단계 · 3단계 조각 4 를 여기서).

지금 토크나이저(tokenizer.py)는 글자 하나 = 번호 하나라, 번호 표에 없는 글자(hello 의 h · 😀)는 KeyError 로 거부한다.
BPE(Byte Pair Encoding — 자주 붙어 나오는 두 조각을 새 번호 하나로 합치기를 되풀이하는 방식)는
가장 작은 조각인 바이트 256종에서 출발해, 자주 나오는 묶음을 하나씩 합쳐 번호 표를 키운다.
바이트로 못 나타내는 글자는 없으니 거부도 없다.

다 만든 뒤 tokenizer.py 의 encode · decode 를 이 파일 것으로 바꿔 끼운다 (이름 · 사용법은 그대로).
"""

from llm.tokenizer import train_text


# ---- 조각 1: 글 <-> 바이트 번호 ----

def to_bytes(s):
    """문자열 -> 바이트 번호 리스트.  예) "안녕" -> [236, 149, 136, 235, 133, 149]

    컴퓨터는 글자를 UTF-8(글자를 바이트로 적는 약속)으로 저장한다 — 영어는 1바이트 · 한글은 3바이트 · 이모지는 4바이트.
    바이트 하나는 0~255 숫자 하나다. 그래서 길이 L 인 문자열이 L 보다 긴 리스트가 된다.
    """
    # 문자열을 UTF-8 약속으로 바이트 묶음으로 바꾼다 — 아래 from_bytes 의 decode(바이트 -> 글)와 짝
    b = s.encode("utf-8")
    # 결과: "안녕" -> b'\xec\x95\x88\xeb\x85\x95' (바이트 6개)
    # b 는 bytes(바이트 묶음) — 화면엔 b'\xec\x95...' 처럼 보인다.
    # list() 로 감싸면 바이트 하나하나가 0~255 숫자로 꺼내진다
    return list(b)


def from_bytes(ids):
    """바이트 번호 리스트 -> 문자열.  예) [236, 149, 136, 235, 133, 149] -> "안녕"  (to_bytes 의 정반대)"""
    # 0~255 숫자 리스트를 다시 바이트 묶음(bytes)으로 묶는다 — 0~255 밖 숫자가 있으면 ValueError
    b = bytes(ids)
    # 바이트 묶음을 UTF-8 약속으로 읽어 글로 되돌린다.
    # 모델이 만든 번호는 한글 한 글자(3바이트)의 중간에서 끊길 수 있다 — 그대로 읽으면 UnicodeDecodeError 로 멈춘다.
    # errors="replace": 글자를 못 이루는 바이트를 만나면 멈추지 않고 � 표시 하나로 바꿔 넣는다
    return b.decode("utf-8", errors="replace")


# 갔다 오면 제자리여야 한다 — 글자 단위 토크나이저와 같은 시험
if __name__ == "__main__":
    print(to_bytes("안녕"))                        # 결과: [236, 149, 136, 235, 133, 149] — '안' = 236 149 136 · '녕' = 235 133 149
    print(len("안녕"), len(to_bytes("안녕")))      # 결과: 2 6 (한글 한 글자 = 바이트 3개)
    print(to_bytes("hi"), to_bytes("😀"))          # 결과: [104, 105] [240, 159, 152, 128] — 영어 1바이트 · 이모지 4바이트 · 지금 토크나이저가 거부하는 글자도 번호가 생김
    print(from_bytes(to_bytes("안녕")))            # 결과: 안녕 (왕복)
    print(repr(from_bytes(to_bytes("안녕")[:2])))  # 결과: '�' — '안' 의 3바이트 중 2개만 · replace 가 없으면 UnicodeDecodeError
    print(len(train_text), len(to_bytes(train_text)))     # 결과: 413139 947863 (바이트로 쪼개면 약 2.3배 길어짐 — 공백 · 기호는 1바이트라 3배보다 작음)
    print(from_bytes(to_bytes(train_text)) == train_text)  # 결과: True (학습용 글 전체 왕복)

# ---- 직접 실행 결과 (uv run python -m llm.bpe) ----
# [236, 149, 136, 235, 133, 149]
# 2 6
# [104, 105] [240, 159, 152, 128]
# 안녕
# '�'
# 413139 947863
# True
