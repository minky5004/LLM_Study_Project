"""학습 로그로 그래프 그리기 — 11단계 조각 4 (11단계 문서 4-1 ~ 4-3).

9 · 10단계에 학습마다 남긴 로그(checkpoints/logs/*.json — 리포 밖)를 읽어 그림 네 종류를 docs/images/ 에 PNG 로 저장한다.
  1) 단계 비교 (비교마다 한 장 · 6장) — 왼쪽 검증 손실 곡선 · 오른쪽 외운 정도(검증 − 학습). 시드 셋의 평균 선 + 최소 ~ 최대 띠
  2) 여정 요약 — 글자 단위 → BPE → 교체 다섯 단계의 검증 최저(글자당) · 점 = 시드 평균 · 막대 = 시드 범위
  3) 속도 — SDPA 교체 전후의 학습 루프 시간
  4) vocab_size — 표 크기마다 드문 번호 수 · 글자당 토큰 수 (작은 그림 둘 · 축 둘 겹치기 안 함)
다시 학습하지 않는다 — 그래프는 로그에서만 (Mario_RL_Project 와 같은 방식).

실행: uv run python -m llm.plot_logs
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 그린다
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = Path("checkpoints/logs")
OUT_DIR = Path("docs/images")

# 실험 순서 · 그림에 쓸 이름. 로그의 label 이 키 — 10단계는 바로 앞 모델 위에 하나씩 쌓은 누적 교체
STAGES = [
    ("char", "Char-level"),
    ("bpe1024", "BPE 1024"),
    ("rmsnorm", "RMSNorm"),
    ("rope", "RoPE"),
    ("swiglu", "SwiGLU"),
    ("sdpa", "SDPA"),
    ("tying", "weight tying"),
]
NAMES = dict(STAGES)

# 색 — 두 실험을 견줄 때 이전 = 파랑 · 교체 = 주황 (색맹이어도 구분되는 짝 · 범례 글로도 이름을 단다)
BEFORE, AFTER = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    # 라벨은 영어 — matplotlib 기본 글꼴(DejaVu Sans) 그대로 · 어느 PC 에서 다시 그려도 같은 모양
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "axes.axisbelow": True,  # 격자선을 막대 · 선 뒤로
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
})


def load_logs():
    """로그 파일을 읽어 {label: {seed: 로그}} 로 묶는다. 같은 label · seed 가 둘이면 나중 파일을 쓴다 (이름이 시각순)."""
    logs = {}
    for path in sorted(LOG_DIR.glob("*.json")):
        log = json.loads(path.read_text(encoding="utf-8"))
        if "label" not in log:  # 9단계 초기 로그(시드 · 이름 칸이 생기기 전)는 건너뛴다
            continue
        logs.setdefault(log["label"], {})[log["seed"]] = log
    return logs


def per_char_curves(log):
    """로그 하나 -> (걸음, 학습 손실, 검증 손실) — 손실은 글자당으로 바꾼 numpy 배열 (11단계 문서 4-1).

    토큰 하나에 글자가 여럿 든 BPE 와 글자 단위를 같은 잣대로 견주려고 글자당으로 바꾼다.
    """
    steps = np.array([h["step"] for h in log["history"]])
    # 토큰당 손실 → 글자당 손실. 글 전체 손실 = 토큰당 × 토큰 수 · 그걸 글자 수로 나눈다
    # 검증은 val_tokens · val_chars 를, 학습은 train_tokens · train_chars 를 쓴다
    val = np.array([h["val"] for h in log["history"]]) * log["val_tokens"] / log["val_chars"]
    train = np.array([h["train"] for h in log["history"]]) * log["train_tokens"] / log["train_chars"]
    return steps, train, val


def seed_band(rows):
    """시드마다 한 줄씩 쌓은 2차원 배열 (시드 수, 걸음 수) -> (평균, 최소, 최대) — 걸음마다 시드 셋을 묶는다 (11단계 문서 4-2)."""
    rows = np.array(rows)
    # 시드 방향으로 묶는다 — 줄(시드)이 쌓인 방향이 0번 축 · (3, 16) -> (16,)
    return rows.mean(axis=0), rows.min(axis=0), rows.max(axis=0)


def plot_compare(logs, before, after, path):
    """그래프 1 — 이전 모델 vs 교체 모델. 왼쪽 검증 손실 · 오른쪽 외운 정도."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for label, color in ((before, BEFORE), (after, AFTER)):
        vals, gaps = [], []
        for log in logs[label].values():
            steps, train, val = per_char_curves(log)
            vals.append(val)
            # 외운 정도 — 검증 손실이 학습 손실보다 얼마나 높은가 (11단계 문서 4-3)
            gaps.append(val - train)
        # 0번째 걸음(학습 전 · 손실 4 ~ 7)은 빼고 그린다 — 넣으면 세로축이 늘어나 뒤쪽 차이가 안 보인다
        s = steps[1:]
        # 범례의 최저값은 그래프 2 와 같은 값 — 시드마다 최저를 먼저 구하고 평균 (평균 곡선의 최저와는 조금 다르다)
        best = np.mean([log["best_val_per_char"] for log in logs[label].values()])
        for ax, rows in ((left, vals), (right, gaps)):
            mean, lo, hi = seed_band([r[1:] for r in rows])
            ax.fill_between(s, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(s, mean, color=color, label=f"{NAMES[label]} — best {best:.4f}")
        # 평균 곡선이 가장 낮은 걸음에 점 — 언제 최저에 닿는지 보이게
        mean = seed_band([r[1:] for r in vals])[0]
        i = mean.argmin()
        left.plot(s[i], mean[i], "o", color=color, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    left.set_title("Validation loss (per char, lower is better)", color=INK, fontsize=11, loc="left")
    right.set_title("Overfitting gap (val − train, higher = more memorized)", color=INK, fontsize=11, loc="left")
    for ax in (left, right):
        ax.set_xlabel("Training step")
    left.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(f"{NAMES[before]} → {NAMES[after]}", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_journey(logs, path):
    """그래프 2 — 단계마다 검증 최저(글자당). 점 = 시드 평균 · 세로 막대 = 시드 최소 ~ 최대."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = range(len(STAGES))
    means, prev = [], None
    for x, (label, name) in zip(xs, STAGES):
        best = np.array([log["best_val_per_char"] for log in logs[label].values()])
        m = best.mean()
        means.append(m)
        ax.vlines(x, best.min(), best.max(), color=BEFORE, linewidth=2)
        ax.plot(x, m, "o", color=BEFORE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        # 값 · 바로 앞 단계 대비 % (첫 점은 기준이라 값만)
        text = f"{m:.4f}" if prev is None else f"{m:.4f}\n{(m - prev) / prev * 100:+.2f}%"
        ax.annotate(text, (x, best.max()), xytext=(0, 8), textcoords="offset points", ha="center",
                    va="bottom", color=INK, fontsize=9)
        prev = m
    ax.plot(xs, means, color=GRID, linewidth=1.5, zorder=1)  # 흐름만 보이게 옅은 선
    ax.set_xticks(list(xs), [name for _, name in STAGES])
    ax.set_ylabel("Best validation loss (per char)")
    lo = min(min(log["best_val_per_char"] for log in logs[l].values()) for l, _ in STAGES)
    hi = max(max(log["best_val_per_char"] for log in logs[l].values()) for l, _ in STAGES)
    ax.set_ylim(lo - 0.01, hi + 0.025)  # 위쪽은 값 글자 자리
    ax.set_title("Best validation loss by stage — dot = mean of 3 seeds · bar = seed range · % = vs previous stage",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speed(logs, path):
    """그래프 3 — SDPA 교체 전후 학습 루프 시간. 이전(SwiGLU)은 같은 조건에서 다시 잰 1번(swiglu-retime)."""
    before = [log["seconds"] for log in logs["swiglu-retime"].values()]
    after = [log["seconds"] for log in logs["sdpa"].values()]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for y, (name, secs, color) in enumerate((("SwiGLU (manual attention)", before, BEFORE),
                                             ("SDPA (built-in attention)", after, AFTER))):
        m = np.mean(secs)
        ax.barh(y, m, height=0.45, color=color)
        if len(secs) > 1:  # 시드가 여럿이면 범위 막대
            ax.hlines(y, min(secs), max(secs), color=INK2, linewidth=1.5)
        ax.annotate(f"{m:.1f}s" + (f" (mean of {len(secs)} seeds)" if len(secs) > 1 else " (1 run)"),
                    (max(secs), y), xytext=(10, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yticks([0, 1], ["SwiGLU\n(manual attention)", "SDPA\n(built-in attention)"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(before + after) * 1.5)
    ax.set_xlabel("Training loop time (s, 5000 steps)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Training time before / after SDPA", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_vocab(path):
    """그래프 4 — vocab_size 고르기 (9단계 조각 6 결과 파일). 드문 번호 수 · 글자당 토큰을 작은 그림 둘로."""
    rows = []
    for line in (LOG_DIR / "9-bpe-vocab-sizes.txt").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.split("|")]
        # 표 줄만: 이름 | vocab | 학습용 토큰 | 글자당 | 드문 번호 | 파라미터 — 첫 칸이 글자 · BPE 인 줄
        if len(cells) == 6 and cells[0] in ("글자", "BPE"):
            name = "Char" if cells[0] == "글자" else cells[0]  # 결과 파일의 "글자" 는 그림에선 영어로
            rows.append((f"{name} {cells[1]}", float(cells[3]), int(cells[4])))
    names = [r[0] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, k, title in ((a, 2, "Rare tokens (< 10 times in training text, lower is better)"),
                         (b, 1, "Tokens per char (lower = shorter sequences)")):
        vals = [r[k] for r in rows]
        colors = [AFTER if n == "BPE 1024" else BEFORE for n in names]  # 고른 크기만 주황
        ax.bar(names, vals, width=0.5, color=colors)
        for x, v in enumerate(vals):
            ax.annotate(f"{v:.3f}" if k == 1 else f"{v}", (x, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", color=INK, fontsize=9)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.15)
    fig.suptitle("Choosing vocab_size — orange = chosen size 1024", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logs = load_logs()

    # 그래프 1 — 이웃한 두 단계마다 한 장 (글자 → BPE · BPE → RMSNorm · ... · SDPA → tying)
    for i, ((before, _), (after, _)) in enumerate(zip(STAGES, STAGES[1:]), start=1):
        plot_compare(logs, before, after, OUT_DIR / f"compare-{i}-{after}.png")
    plot_journey(logs, OUT_DIR / "journey.png")
    plot_speed(logs, OUT_DIR / "speed-sdpa.png")
    plot_vocab(OUT_DIR / "vocab-size.png")
    print(sorted(p.name for p in OUT_DIR.glob("*.png")))

# ---- 직접 실행 결과 (uv run python -m llm.plot_logs · 몇 초) ----
# ['compare-1-bpe1024.png', 'compare-2-rmsnorm.png', 'compare-3-rope.png', 'compare-4-swiglu.png', 'compare-5-sdpa.png',
#  'compare-6-tying.png', 'journey.png', 'speed-sdpa.png', 'vocab-size.png']
# 여정 요약 (검증 최저 · 글자당 · 시드 3개 평균): 글자 단위 1.6566 → BPE 1024 1.7074 (+3.07%) → RMSNorm 1.7119 (+0.26%)
#   → RoPE 1.7028 (-0.53%) → SwiGLU 1.6870 (-0.93%) → SDPA 1.6870 (+0.00%) → weight tying 1.6583 (-1.70%)
# 속도: SwiGLU 312.9초 (1번) → SDPA 286.4초 (시드 3개 평균)
