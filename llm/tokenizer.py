"""텍스트 <-> 토큰(정수) 변환 — 9단계 조각 7 부터 BPE 1024 (결정 기록 0016 · 0017).

8단계까지는 글자 하나 = 번호 하나였다 (그 코드는 char_tokenizer.py 에 그대로 남아 있다).
이제 llm/bpe.py 가 만든 합치기 표로 바꾼다. 바깥에서 쓰는 이름 · 모양은 그대로다 (9단계 문서 7-1):
  encode(글) -> 번호 리스트 · decode(번호 리스트) -> 글 · vocab_size · train_text · val_text
그래서 이 파일을 가져다 쓰는 dataset.py · train.py · generate.py 는 고칠 것이 없다.

합치기 표 파일(checkpoints/bpe_merges.json)이 먼저 있어야 한다 — 없으면 `uv run python -m llm.bpe_vocab` (약 12분).
"""

from pathlib import Path

# bpe.py 의 encode · decode 에 별명을 붙여 가져온다 (9단계 문서 7-2).
# 아래에서 같은 이름 encode · decode 를 새로 만드는데, 별명이 없으면 그 안에서 부른 encode 가 자기 자신이 되어
# 끝없이 자기를 부르다 RecursionError 로 멈춘다.
from llm.bpe import decode as bpe_decode
from llm.bpe import encode as bpe_encode
from llm.bpe import load

# 학습용 · 검증용 글은 글자 단위 파일이 읽어 둔 것을 그대로 가져와 다시 내보낸다 — dataset.py 가 여기서 가져간다
from llm.char_tokenizer import train_text, val_text

# 합치기 표 파일과 쓸 크기. 파일에는 4096 표가 있고, 앞부분만 잘라 쓴다 (조각 6 · 문서 6-1)
MERGES_PATH = Path("checkpoints/bpe_merges.json")
VOCAB_SIZE = 1024  # 결정 기록 0016 — 셋 중 드문 번호가 가장 적음

# {(앞 번호, 뒤 번호): 새 번호} 사전 — 앞 768줄 (1024 - 256)
merges = load(MERGES_PATH, VOCAB_SIZE)

# 번호 표 전체 크기 = 모델이 만들 번호 칸 수 (9단계 문서 7-4).
# 1024 라고 손으로 적지 않고 불러온 표에서 센다 — VOCAB_SIZE 를 바꿔도 이 줄은 그대로.
# 바이트 번호 256개(0~255) + 합치기 표 한 줄마다 새 번호 하나
vocab_size = len(merges) + 256
# 결과: 1024 (768줄 + 256)


def encode(s):
    """문자열 -> 번호 리스트.  예) "사용자:" -> [270, 58]

    글자 단위 때와 달리 길이 L 인 글이 L 보다 짧은 리스트가 된다 (검증용 글 기준 글자당 0.743 토큰).
    어떤 글자든 바이트로 쪼갤 수 있어 KeyError 로 거부하는 일이 없다.
    """
    # bpe.py 의 encode 는 (글, 합치기 표) 두 개를 받는다. 바깥에서는 encode(글) 하나만 넘기니 표는 여기서 채워 넘긴다
    # 받은 글 s 는 그대로 · 표는 merges 사전 자체를 (merges[s] 는 글을 키로 찾아 KeyError · VOCAB_SIZE 는 숫자라 표가 아님)
    return bpe_encode(s, merges)


def decode(ids):
    """번호 리스트 -> 문자열.  예) [270, 58] -> "사용자:"  (encode 의 정반대)"""
    # encode 와 같은 모양 — 표는 여기서 채워 넘긴다. 글자 중간에서 끊긴 번호는 � 로 보인다 (조각 5)
    return bpe_decode(ids, merges)


# 갔다 오면 제자리여야 한다 — 글자 단위 때와 같은 시험 (3단계)
if __name__ == "__main__":
    print(vocab_size)                                 # 결과: 1024
    print(encode("사용자:"), decode(encode("사용자:")))  # 결과: [270, 58] 사용자: — '사용자' 가 번호 하나 (글자 단위 때는 4개)
    print(encode("\n"))  # 줄바꿈 번호 — generate.py 가 멈출 번호로 쓴다 (9단계 문서 7-6)
    # 결과: [10] — 줄바꿈의 바이트 번호 그대로 (글자 단위 때는 [0])
    print(encode("hello 😀"))  # 글자 단위 때는 KeyError 로 거부하던 글
    # 결과: [104, 101, 108, 108, 111, 32, 240, 159, 152, 128] — 거부 없음 · 학습용 글에 영어가 드물어 합쳐진 것 없이 바이트 그대로
    print(decode(encode(val_text)) == val_text)       # 결과: True (검증용 글 전체 왕복)

# ---- 직접 실행 결과 (uv run python -m llm.tokenizer · 약 5초) ----
# 1024
# [270, 58] 사용자:
# [10]
# [104, 101, 108, 108, 111, 32, 240, 159, 152, 128]
# True
