"""직접 짠 BPE(9단계) 와 `tokenizers` 라이브러리 BPE 견주기 — 11단계 조각 3 (11단계 문서 3-1 ~ 3-3).

같은 학습용 글 · 같은 크기 1024 · 둘 다 바이트 기준으로 학습하고, 넷을 잰다:
  학습 시간 · 검증용 글자당 토큰 수 · 왕복(decode(encode(글)) == 글) · 두 표에 겹친 조각 수
그리고 같은 문장을 둘이 어떻게 자르는지 나란히 본다.

실행: uv run --with tokenizers python -m llm.compare_tokenizers
  --with 는 이 실행 한 번에만 라이브러리를 빌려 쓴다 — pyproject.toml 은 안 바뀐다 (비교에만 쓰고 의존성에는 안 넣음)
  우리 BPE 를 1024 까지 새로 학습하느라 몇 분 걸린다 (합치기 표 파일을 불러 쓰지 않고 시간을 재려고 다시 배운다)
"""

import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from llm import bpe
from llm.char_tokenizer import train_text, val_text

VOCAB_SIZE = 1024  # 9단계 결정 기록 0016 과 같은 크기


def per_char(ids):
    """검증용 글자당 토큰 수 — 토큰 수 ÷ 글자 수 (9단계 문서 6-2). 작을수록 토큰 하나에 글자가 많이 든다."""
    # 토큰 수(ids 의 길이) ÷ 검증용 글의 글자 수 — set(ids) 로 세면 겹친 번호가 합쳐져 "번호 종류 수" 가 된다
    return len(ids) / len(val_text)


def byte_map():
    """라이브러리 조각 글 -> 바이트로 되돌릴 표.  예) 'Ġ' -> 32 (공백)

    라이브러리(GPT-2 방식)는 바이트 256개를 눈에 보이는 글자 256개로 바꿔 조각을 글로 저장한다.
    공백(32) · 줄바꿈(10) 처럼 안 보이는 바이트는 256 뒤의 글자로 옮긴다 — 공백이 'Ġ' 로 보이는 이유.
    우리 조각(바이트 묶음)과 견주려면 이 바꾸기를 거꾸로 되돌려야 한다.
    """
    # 눈에 보이는 바이트(! ~ ~ · ¡ ~ ¬ · ® ~ ÿ)는 자기 글자 그대로
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    # 나머지 바이트(공백 · 줄바꿈 등)는 256, 257, ... 번 글자로
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # {글자: 바이트} — 라이브러리 조각 글을 글자마다 바이트로 되돌린다
    return {chr(c): b for b, c in zip(bs, cs)}


def show(pieces):
    """바이트 묶음 리스트 -> 보기 좋은 글 리스트. 한글 한 글자 중간에서 끊긴 조각은 � 로 보인다."""
    return [p.decode("utf-8", errors="replace") for p in pieces]


if __name__ == "__main__":
    # ---- 1) 우리 BPE 학습 (9단계 bpe.train) ----
    t = time.perf_counter()
    ours_merges, _ = bpe.train(train_text, VOCAB_SIZE)
    ours_sec = time.perf_counter() - t

    # ---- 2) 라이브러리 BPE 학습 ----
    # 빈 BPE 모델을 만들고, 먼저 자르기(pre_tokenizer) · 되돌리기(decoder) 를 바이트 기준(ByteLevel)으로 단다
    # add_prefix_space=False: 글 맨 앞에 공백을 몰래 붙이지 않는다 — 우리 BPE 도 안 붙여서 같은 글을 보게
    hf = Tokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf.decoder = decoders.ByteLevel()
    # initial_alphabet: 바이트 256개를 처음부터 표에 넣는다 — 우리 BPE 의 0 ~ 255 번과 같은 출발점
    # (안 넣으면 학습용 글에 나온 바이트만 들어가 영어 · 이모지 바이트가 빠질 수 있다)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    t = time.perf_counter()
    hf.train_from_iterator([train_text], trainer)  # 학습용 글 하나를 통째로 넘긴다
    hf_sec = time.perf_counter() - t

    # ---- 3) 검증용 글로 재기 ----
    ours_ids = bpe.encode(val_text, ours_merges)
    hf_ids = hf.encode(val_text).ids  # encode 는 여러 정보를 담은 묶음을 돌려준다 — .ids 가 번호 리스트

    print(f"학습 시간    우리 {ours_sec:.1f}초 · 라이브러리 {hf_sec:.2f}초")
    print(f"글자당 토큰  우리 {per_char(ours_ids):.4f} · 라이브러리 {per_char(hf_ids):.4f}")
    print(f"왕복         우리 {bpe.decode(ours_ids, ours_merges) == val_text} · 라이브러리 {hf.decode(hf_ids) == val_text}")

    # ---- 4) 두 표에 겹친 조각 (문서 3-3) ----
    # 우리 표: 번호 -> 바이트 묶음 (bpe.decode 안에서 만드는 표와 같은 계산 — 9단계 문서 5-2)
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in ours_merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    ours_set = set(vocab.values())

    # 라이브러리 표: 조각 글 -> 바이트 묶음 (byte_map 으로 글자마다 되돌림)
    m = byte_map()
    hf_set = {bytes(m[c] for c in piece) for piece in hf.get_vocab()}

    # 두 set 에 다 있는 조각만 남긴다 — & 는 교집합 (문서 3-3 ②)
    both = ours_set & hf_set
    print(f"겹친 조각    {len(both)} / {VOCAB_SIZE}  (우리 {len(ours_set)} · 라이브러리 {len(hf_set)})")

    # ---- 5) 같은 문장을 어떻게 자르나 ----
    for s in ["사용자: 배고파", "오늘 뭐해", "hello 😀"]:
        ours_pieces = [vocab[i] for i in bpe.encode(s, ours_merges)]
        hf_pieces = [bytes(m[c] for c in piece) for piece in hf.encode(s).tokens]  # .tokens 는 조각 글 리스트
        print(repr(s))
        print("  우리      ", show(ours_pieces))
        print("  라이브러리", show(hf_pieces))

# ---- 직접 실행 결과 (uv run --with tokenizers python -m llm.compare_tokenizers · tokenizers 0.23.3 · 약 2분) ----
# 학습 시간    우리 113.1초 · 라이브러리 0.40초
# 글자당 토큰  우리 0.7434 · 라이브러리 0.7434
# 왕복         우리 True · 라이브러리 True
# 겹친 조각    1022 / 1024  (우리 1024 · 라이브러리 1024)
# '사용자: 배고파'
#   우리       ['사용자', ':', ' 배', '고', '�', '�']
#   라이브러리 ['사용자', ':', ' 배', '고', '�', '�']
# '오늘 뭐해'
#   우리       ['�', '��', '�', '�', ' 뭐', '해']
#   라이브러리 ['�', '��', '�', '�', ' 뭐', '해']
# 'hello 😀'
#   우리       ['h', 'e', 'l', 'l', 'o', ' ', '�', '�', '�', '�']
#   라이브러리 ['h', 'e', 'l', 'l', 'o', ' ', '�', '�', '�', '�']
#   글자당 토큰 소수 넷째 자리까지 같음 · 세 문장 자르기 전부 같음 · 학습은 약 280배 차이
#   다른 조각 두 개 (따로 확인 · 바이트 16진수로 적음):
#     우리만       932번 eb a6 (한 글자의 앞 두 바이트) · 1023번 20 eb 8d (공백 + 한 글자의 앞 두 바이트)
#     라이브러리만 929번 a6 b4 (한 글자의 뒤 두 바이트) · 1023번 eb 8c 80 ('대' 한 글자)
#     — 둘 다 900번대 이후 · 1등 쌍이 동점일 때 고르는 방법이 달라 갈린 것으로 봄 (표 뒤쪽일수록 쌍 횟수가 비슷해 동점이 잦다)
