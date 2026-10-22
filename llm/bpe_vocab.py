"""BPE 번호 표 크기(vocab_size) 고르기 — 9단계 조각 6.

4096 합치기 표를 한 번만 배워 파일로 저장하고, 앞부분을 잘라 1024 · 2048 · 4096 을 같은 방법으로 잰다 (문서 6-1).
지금 글자 단위 토크나이저(글자 1,244종)도 같은 잣대로 재서 나란히 놓는다.

실행: uv run python -m llm.bpe_vocab
  처음엔 표를 배우느라 십 분 넘게 걸린다. 표 파일이 생긴 뒤에는 배우기를 건너뛰고 재기만 한다.
"""

import time
from collections import Counter
from pathlib import Path

from llm import tokenizer
from llm.bpe import decode, encode, load, save, train
from llm.config import ModelConfig
from llm.model import GPT
from llm.tokenizer import train_text, val_text

# 합치기 표 파일 — 모델 파일(model.pt)처럼 데이터에서 다시 만들 수 있는 결과물이라 checkpoints/ 에 (리포 밖)
MERGES_PATH = Path("checkpoints/bpe_merges.json")
SIZES = [1024, 2048, 4096]


def count_params(vocab_size):
    """8단계 모델 크기(6층 · 256칸)로 모델을 만들어 파라미터 수를 센다 (문서 6-4) — train.py 와 같은 설정."""
    config = ModelConfig(vocab_size=vocab_size, n_layer=6, n_embd=256, n_head=8, dropout=0.2)
    return sum(p.numel() for p in GPT(config).parameters())


def report(name, vocab_size, train_ids, val_ids):
    """잣대 넷을 한 줄로 — 학습용 글 토큰 수 · 검증용 글자당 토큰 · 드문 번호 수 · 파라미터 수."""
    # 번호마다 학습용 글에 몇 번 나왔나 — Counter 는 한 번도 안 나온 번호를 물으면 0 을 돌려준다
    counts = Counter(train_ids)
    # 0 ~ vocab_size-1 번 중 10번도 안 나온 번호 수 (문서 6-3) — 한 번도 안 나온 번호(0번)도 여기 든다
    rare = sum(1 for i in range(vocab_size) if counts[i] < 10)

    # 검증용 글자당 토큰 (문서 6-2) — 토큰 수 ÷ 글자 수. 글자 단위는 정의상 1.0 · 작을수록 토큰 하나에 글자가 많이 든다
    # 분모는 encode 전의 글(val_text) 길이 — 학습용 토큰 수 ÷ 검증용 토큰 수는 두 글의 크기 비율(약 9)이 될 뿐
    per_char = len(val_ids) / len(val_text)

    print(f"{name:>6} | {vocab_size:>5} | {len(train_ids):>9,} | {per_char:.3f} | {rare:>5} | {count_params(vocab_size):>10,}")


if __name__ == "__main__":
    # 1) 4096 표를 한 번 배워 저장 — 파일이 이미 있으면 건너뛴다 (배우기만 십 분 넘게 걸려서)
    if not MERGES_PATH.exists():
        t = time.perf_counter()
        merges, _ = train(train_text, max(SIZES))
        print(f"4096 표 배우기 {time.perf_counter() - t:.0f}초")
        MERGES_PATH.parent.mkdir(exist_ok=True)  # checkpoints/ 가 없으면 만든다
        save(merges, MERGES_PATH)

    # 2) 잣대 넷을 나란히 — 맨 윗줄은 지금 글자 단위 토크나이저
    print("  이름 | vocab | 학습용 토큰 | 글자당 | 드문 번호 | 파라미터")
    report("글자", tokenizer.vocab_size, tokenizer.encode(train_text), tokenizer.encode(val_text))
    for size in SIZES:
        merges = load(MERGES_PATH, size)  # 4096 표의 앞부분만 (문서 6-1)
        report("BPE", size, encode(train_text, merges), encode(val_text, merges))

    # 3) 같은 글이 표 크기마다 어떻게 쪼개지나 — 번호 하나씩 되돌려 보면 조각 경계가 보인다
    #    (한글 한 글자 중간에서 끊긴 조각은 � 로 보인다)
    for size in SIZES:
        merges = load(MERGES_PATH, size)
        for s in ["오늘 뭐해", "배고파"]:
            print(size, [decode([i], merges) for i in encode(s, merges)])

# ---- 직접 실행 결과 (uv run python -m llm.bpe_vocab · 조각 6.5 낱말 단위로 먼저 자르기 뒤 · 전체 약 12분) ----
# 4096 표 배우기 331초
#   이름 | vocab | 학습용 토큰 | 글자당 | 드문 번호 | 파라미터
#     글자 |  1244 |   413,139 | 1.000 |   531 |  5,410,012
#    BPE |  1024 |   307,793 | 0.743 |   180 |  5,297,152
#    BPE |  2048 |   253,257 | 0.613 |   270 |  5,822,464
#    BPE |  4096 |   219,978 | 0.537 |   669 |  6,873,088
# 1024 ['�', '��', '�', '�', ' 뭐', '해']
# 1024 ['�', '�', '고', '�', '�']
# 2048 ['오�', '��', ' 뭐', '해']
# 2048 ['�', '�', '고', '파']
# 4096 ['오�', '��', ' 뭐', '해']
# 4096 ['�', '�', '고', '파']
# <- 조각 6.5 전보다 토큰이 많다 (1024: 244,885 -> 307,793) — 조각 경계를 넘는 합치기('요.\n\n사용자: �' 같은 덩어리)를 막아서
#    드문 번호는 1024 가 여전히 가장 적고(180) 4096 은 글자 단위(531)보다도 많아짐(669)
#    맨 앞에 공백 없이 놓인 '오늘' · '배고파' 는 여전히 바이트 조각 — 학습용 글에선 늘 ': ' 뒤 ' 오늘' · ' 배고파' 처럼 공백과 붙어 나와서.
#    실제로 모델에 넣는 시작 글로는 잘 잘린다 (1024): '사용자: 오늘 뭐해\n봇:' -> ['사용자', ':', ' 오늘', ' 뭐', '해', '\n', '봇', ':']
#    조각 6.5 전에는 '사용자' 가 학습용 글에 혼자 1번 · 줄바꿈(10) 혼자 0번 -> 지금은 10,637번 · 31,910번 (1024 표 · 따로 잼)
