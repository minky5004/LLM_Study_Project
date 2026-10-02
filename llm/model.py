"""트랜스포머 아키텍처: 임베딩 -> attention -> FFN -> 조립.

GPT-2식으로 먼저 완성하고, 돌아가면 부품을 하나씩 현대식으로 교체한다 (4단계 문서 참고).
지금은 조각 1: 번호를 벡터로 바꾸는 토큰 임베딩만 있다.
"""

import torch
import torch.nn as nn

from llm.config import ModelConfig


# ---- 조각 1: 토큰 임베딩 (번호 -> 벡터) ----

class GPT(nn.Module):
    """모델 전체. 지금은 부품이 토큰 임베딩 하나뿐이고 조각이 늘 때마다 부품을 붙인다.

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

    def forward(self, idx):
        """idx: 번호 묶음. shape (B, T) — B = 문장 몇 개를 한꺼번에, T = 문장당 토큰 몇 개.
        반환: 번호마다 벡터가 붙은 텐서. shape (B, T, n_embd).
        """
        # 번호 하나하나를 표에서 찾아 그 줄(벡터)로 바꾼다. 모양만 (B, T) -> (B, T, n_embd) 로 한 칸 늘어난다.
        tok_emb = self.wte(idx)
        return tok_emb


# 직접 실행했을 때만 확인용 출력
if __name__ == "__main__":
    config = ModelConfig()
    model = GPT(config)

    B, T = 2, 5  # 문장 2개, 문장당 토큰 5개
    # 가짜 입력: 0 이상 상한 미만의 무작위 번호. 번호는 진짜 vocab 안의 값이어야 표에 줄이 있다.
    idx = torch.randint(0, config.vocab_size, (B, T))

    out = model(idx)
    print(idx.shape)   # 기대: torch.Size([2, 5])
    print(out.shape)   # 기대: torch.Size([2, 5, 128])
    print(sum(p.numel() for p in model.parameters()))  # 기대: 8832 (69 x 128, 표 안의 숫자 개수)
