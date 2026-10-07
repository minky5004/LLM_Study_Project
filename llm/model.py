"""트랜스포머 아키텍처: 임베딩 -> attention -> FFN -> 조립.

GPT-2식으로 먼저 완성하고, 돌아가면 부품을 하나씩 현대식으로 교체한다 (4단계 문서 참고).
지금은 조각 2까지: 토큰 임베딩 + 위치 임베딩 (몇 번째 글자인지 알려 주기).
"""

import torch
import torch.nn as nn

from llm.config import ModelConfig


# ---- 조각 1: 토큰 임베딩 (글자 번호 -> 숫자 줄) · 조각 2: 위치 임베딩 (순서 알려 주기) ----

class GPT(nn.Module):
    """모델 전체를 담는 상자. 지금은 안에 표 두 개뿐이고, 조각이 늘 때마다 부품을 붙인다.

    nn.Module 을 물려받는 이유: 나중에 학습할 때 "고쳐야 할 숫자 전부" 를 한꺼번에 찾아야 하는데,
    이 상자 안에 만든 표 · 가중치를 torch 가 자동으로 모아 주기 때문이다.
    """

    def __init__(self, config: ModelConfig):
        # __init__ = 부품을 만들어 두는 곳. 모델을 만들 때 딱 한 번 실행된다.
        # (config 에는 vocab_size · n_embd 같은 크기 값이 들어 있다 — 크기를 바꾸려면 config.py 만 고치면 된다)

        # 부모(nn.Module)의 준비 작업을 먼저 실행한다.
        # 부모가 "부품 목록 장부" 를 만드는 곳이라, 이 줄을 빼먹으면 아래에서 만든 표가 장부에 안 적혀서 학습 때 못 찾는다.
        super().__init__()

        # 받은 크기 값을 상자에 보관해 둔다. 나중에 다른 곳에서 self.config 로 꺼내 쓰려고.
        self.config = config

        # 토큰 임베딩 표: 글자 번호 하나 -> 숫자 n_embd 개짜리 줄 하나를 찾아 주는 표.
        # 표의 모양은 (줄 수, 칸 수).
        #   - 줄 수: 번호가 곧 줄 번호라서, 모든 번호에 줄이 있으려면 "글자 종류 수" 와 같아야 한다.
        #   - 칸 수: 글자 하나를 숫자 몇 개로 설명할지 = n_embd (128).
        # 처음엔 아무 숫자로 채워지고, 학습하면서 조금씩 고쳐진다.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # 결과: self.wte.weight.shape = torch.Size([69, 128]) — 줄 69개(글자 종류) × 칸 128개

        # 위치 임베딩 표: 자리 번호 하나 -> 숫자 n_embd 개짜리 줄 하나. 글자 표와 같은 nn.Embedding 이다.
        # 같은 글자는 어느 자리에 있든 글자 표에서 같은 줄을 받는다. 글은 순서가 뜻이라 자리 정보를 따로 넣어 줘야 한다.
        #   - 줄 수: 자리가 몇 종류냐 = 모델이 한 번에 볼 수 있는 글자 수의 최댓값 = block_size (T 가 아니다 — T 는 입력마다 달라서 __init__ 시점엔 모른다)
        #   - 칸 수: 글자 줄에 "더할" 거라 글자 표와 칸 수가 같아야 한다 (이어 붙이면 줄이 길어져 뒤 부품이 전부 바뀐다)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # 결과: self.wpe.weight.shape = torch.Size([128, 128]) — 줄 128개(자리) × 칸 128개

    def forward(self, idx):
        """forward = 데이터가 실제로 흘러가는 곳. 모델을 부를 때마다 실행된다.

        idx: 글자 번호 묶음. shape (B, T) — B = 문장 몇 개를 한꺼번에, T = 문장 하나의 글자 몇 개.
        반환: 번호마다 숫자 줄이 붙은 텐서. shape (B, T, n_embd).
        """
        # 번호 묶음을 표에 넣으면, 번호마다 그 번호의 줄이 꺼내져 나온다 (계산이 아니라 표 찾기).
        # 번호 하나가 숫자 n_embd 개로 바뀌므로 shape 맨 뒤에 n_embd 가 붙는다: (B, T) -> (B, T, n_embd)
        tok_emb = self.wte(idx)
        # 결과: idx (2, 5) -> tok_emb (2, 5, 128). 같은 번호는 같은 줄을 받는다 (번호 [0, 1, 0] 의 첫째 · 셋째 줄이 똑같음 — 실행 확인)

        # 이번 입력의 글자 수 T 를 shape 에서 꺼낸다. (B, T) 의 뒤쪽 값. shape 는 값 둘짜리 묶음이라 B, T = ... 로 풀어 받을 수 있다.
        B, T = idx.shape

        # 자리 번호 [0, 1, ..., T-1] 을 직접 만든다. 자리는 입력 값이 아니라 "몇 번째 칸이냐" 일 뿐이라 문장이 몇 개든 같다.
        # device=idx.device: 입력이 GPU 에 있으면 자리 번호도 같은 곳에 만든다 (CPU/GPU 가 섞이면 에러)
        # 자리 번호는 글자 개수 T 만큼 필요하다 (문장 개수 B 와는 무관).
        pos = torch.arange(T, device=idx.device)
        # 결과: T = 5 일 때 pos = tensor([0, 1, 2, 3, 4]) — shape (5,)

        # 자리 번호로 자리 줄을 꺼낸다. shape: (T,) -> (T, n_embd)  (문장 개수 B 가 없다 — 자리 줄은 어느 문장이든 같아서 한 벌이면 된다)
        pos_emb = self.wpe(pos)
        # 결과: pos (5,) -> pos_emb (5, 128)

        # 글자 줄과 자리 줄을 합친다. (B, T, n_embd) 와 (T, n_embd) 는 모양이 다르지만
        # broadcasting 이 pos_emb 를 B 개 문장에 똑같이 복사해 맞춰 준다. 결과 shape: (B, T, n_embd)
        # 이어 붙이는 게 아니라 칸마다 더한다 (그래서 shape 가 그대로).
        x = tok_emb + pos_emb
        # 결과: (2, 5, 128) + (5, 128) -> (2, 5, 128). 번호 [0, 1, 0] 의 첫째 · 셋째 줄이 이제 서로 다름 (조각 1 에서는 같았다 — 실행 확인)
        return x


# 직접 실행했을 때만 확인용 출력
if __name__ == "__main__":
    config = ModelConfig()
    model = GPT(config)

    B, T = 2, 5  # 문장 2개, 문장당 글자 5개
    # 가짜 입력: 0 이상 상한 미만의 무작위 번호. 번호는 진짜 글자 번호 범위 안이어야 표에 줄이 있다.
    # 상한이 표의 줄 수보다 크면 어쩌다 한 번씩 IndexError 가 나서 "실행마다 터졌다 안 터졌다" 하는 버그가 된다.
    # 그래서 상한은 표의 줄 수와 같은 vocab_size (config.n_embd 는 줄 수가 아니라 칸 수라 안 된다).
    idx = torch.randint(0, config.vocab_size, (B, T))

    out = model(idx)
    print(idx.shape)   # 기대: torch.Size([2, 5])
    print(out.shape)   # 기대: torch.Size([2, 5, 128]) (자리를 더해도 모양은 그대로)
    # 기대: 25216 = 글자 표 8832 (69 x 128) + 자리표 16384 (128 x 128)
    print(sum(p.numel() for p in model.parameters()))

# ---- 직접 실행 결과 (uv run python -m llm.model) ----
# torch.Size([2, 5])
# torch.Size([2, 5, 128])
# 25216
