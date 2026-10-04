"""트랜스포머 아키텍처: 임베딩 -> attention -> FFN -> 조립.

GPT-2식으로 먼저 완성하고, 돌아가면 부품을 하나씩 현대식으로 교체한다 (4단계 문서 참고).
지금은 조각 3까지: 토큰 임베딩 + 위치 임베딩 + attention 의 Q/K/V.
"""

import torch
import torch.nn as nn

from llm.config import ModelConfig


# ---- 조각 1: 토큰 임베딩 (번호 -> 벡터) · 조각 2: 위치 임베딩 (순서 알려 주기) ----

class GPT(nn.Module):
    """모델 전체. 지금은 부품이 임베딩 표 두 개뿐이고 조각이 늘 때마다 부품을 붙인다.

    nn.Module 을 물려받는 이유: 이 클래스 안에 만든 표 · 가중치(학습으로 고쳐지는 숫자)를
    torch 가 자동으로 찾아 모아 주기 때문이다 (나중에 옵티마이저가 이걸 한꺼번에 고친다).
    """

    def __init__(self, config: ModelConfig):
        # nn.Module 을 쓰는 클래스는 맨 처음에 부모의 __init__ 을 한 번 불러 줘야 한다.
        # (부모가 "부품 목록 장부" 를 만드는 곳이라, 건너뛰면 아래에서 만든 부품이 장부에 안 적힌다)
        super().__init__()
        self.config = config

        # 토큰 임베딩 표: 번호 한 개 -> 길이 n_embd 짜리 벡터 한 개를 꺼내 주는 표.
        # 표의 모양은 (표의 줄 수, 줄 하나의 칸 수). 번호가 줄 번호가 되므로 줄 수는 "번호가 몇 종류냐" 와 같아야 한다.
        # 처음엔 무작위 숫자로 채워지고, 학습하면서 조금씩 고쳐진다.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # 위치 임베딩 표: 위치 번호 한 개 -> 길이 n_embd 짜리 벡터 한 개. 토큰 표와 같은 nn.Embedding 이다.
        # 토큰 벡터에 "더해야" 하므로 줄 하나의 칸 수는 토큰 표와 같아야 하고,
        # 줄 수는 "위치가 몇 종류냐" = 모델이 한 번에 볼 수 있는 토큰 수의 최댓값(block_size)이다.
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

    def forward(self, idx):
        """idx: 번호 묶음. shape (B, T) — B = 문장 몇 개를 한꺼번에, T = 문장당 토큰 몇 개.
        반환: 번호마다 벡터가 붙은 텐서. shape (B, T, n_embd).
        """
        # 번호 하나하나를 표에서 찾아 그 줄(벡터)로 바꾼다. 모양만 (B, T) -> (B, T, n_embd) 로 한 칸 늘어난다.
        tok_emb = self.wte(idx)

        # 이번 입력의 토큰 수 T 를 shape 에서 꺼낸다. (B, T) 의 뒤쪽 값. shape 는 튜플이라 B, T = ... 로 풀어 받을 수 있다.
        B, T = idx.shape

        # 위치 번호 [0, 1, ..., T-1] 을 직접 만든다. 위치는 입력 값이 아니라 "몇 번째 칸이냐" 일 뿐이라 문장이 몇 개든 같다.
        # device=idx.device: 입력이 GPU 에 있으면 위치 번호도 같은 곳에 만든다 (CPU/GPU 가 섞이면 에러)
        # shape: (T,)
        pos = torch.arange(T, device=idx.device)

        # 위치 번호를 표에서 찾아 벡터로 바꾼다. shape: (T,) -> (T, n_embd)
        pos_emb = self.wpe(pos)

        # 토큰 벡터와 위치 벡터를 합친다. (B, T, n_embd) 와 (T, n_embd) 는 모양이 다르지만
        # broadcasting 이 pos_emb 를 B 개 문장에 똑같이 복사해 맞춰 준다. 결과 shape: (B, T, n_embd)
        # 이어 붙이는 게 아니라 칸마다 더한다 (그래서 shape 가 그대로).
        x = tok_emb + pos_emb
        return x


# ---- 조각 3: Q / K / V (한 글자를 질문 · 열쇠 · 값 세 벡터로) ----

class CausalSelfAttention(nn.Module):
    """attention 부품. 지금은 조각 3 이라 Q / K / V 를 만들어 돌려주는 데까지만 한다.

    Q (질문): 이 글자가 앞 글자들 중에서 무엇을 찾고 있는지
    K (열쇠): 이 글자가 다른 글자의 질문에 "나는 이런 글자야" 하고 내미는 표시
    V (값):   이 글자가 누군가에게 참고당할 때 건네줄 정보
    셋 다 같은 글자 벡터 x 에서 만들지만, 변환 기계(nn.Linear)가 셋으로 따로라서 결과가 서로 다르다.
    점수 계산 · 마스킹 · softmax(조각 4) 는 아직 없다.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()  # nn.Module 을 쓰는 클래스의 첫 줄 (GPT 와 같은 이유)

        # nn.Linear(들어오는 칸 수, 나가는 칸 수): 벡터를 받아 새 벡터로 바꿔 주는 기계. 안에 학습으로 고쳐지는 숫자 표(가중치)가 있다.
        # 지금은 모양을 그대로 유지한다 — 입력 (B, T, n_embd) 이 출력도 (B, T, n_embd). (칸을 쪼개는 건 조각 5 multi-head 에서)
        # 질문을 만드는 기계. 들어오는 것이 임베딩을 거친 벡터 x 라서 들어오는 칸 · 나가는 칸 모두 n_embd 다 (vocab_size 는 번호 → 벡터 표의 줄 수일 뿐).
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)

        # 열쇠를 만드는 기계. 질문 기계와 크기는 같지만 숫자 표가 따로라서 학습하면 서로 다른 일을 하게 된다.
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)

        # 값을 만드는 기계. 마찬가지로 숫자 표가 따로다.
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        """x: 위치까지 섞인 글자 벡터. shape (B, T, n_embd).
        반환: (q, k, v) 세 텐서. 각각 shape (B, T, n_embd). (조각 4 에서 이어서 점수 계산에 쓴다)
        """
        # Linear 는 입력의 맨 뒤 칸(n_embd)에만 적용된다 — 글자마다 따로, 같은 변환을 한다.
        # 질문: q_proj 에 x 를 넣는다. shape: (B, T, n_embd)
        q = self.q_proj(x)

        # 열쇠 (위 q 와 같은 모양으로 k_proj 에 x 를 넣는다)
        k = self.k_proj(x)

        # 값: v_proj 에 x 를 넣는다. shape: (B, T, n_embd)
        v = self.v_proj(x)
        return q, k, v


# 직접 실행했을 때만 확인용 출력
if __name__ == "__main__":
    config = ModelConfig()
    model = GPT(config)

    B, T = 2, 5  # 문장 2개, 문장당 토큰 5개
    # 가짜 입력: 0 이상 상한 미만의 무작위 번호. 번호는 진짜 vocab 안의 값이어야 표에 줄이 있다.
    idx = torch.randint(0, config.vocab_size, (B, T))

    out = model(idx)
    print(idx.shape)   # 기대: torch.Size([2, 5])
    print(out.shape)   # 기대: torch.Size([2, 5, 128]) (위치를 더해도 모양은 그대로)
    # 기대: 25216 = 토큰 표 8832 (69 x 128) + 위치 표 16384 (128 x 128)
    print(sum(p.numel() for p in model.parameters()))

    # 조각 3 확인: 임베딩 결과 out 을 attention 부품에 넣어 Q / K / V 를 만든다
    attn = CausalSelfAttention(config)
    q, k, v = attn(out)
    print(q.shape, k.shape, v.shape)   # 기대: 셋 다 torch.Size([2, 5, 128]) (모양은 그대로)
    # 기대: 49536 = (128 x 128 가중치 + 128 편향) x 3
    print(sum(p.numel() for p in attn.parameters()))
