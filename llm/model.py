"""트랜스포머 아키텍처: 임베딩 -> attention -> FFN -> 조립.

GPT-2식으로 먼저 완성하고, 돌아가면 부품을 하나씩 현대식으로 교체한다 (4단계 문서 참고).
지금은 조각 4까지: 토큰 임베딩 + 위치 임베딩 (몇 번째 글자인지 알려 주기) + attention (Q/K/V -> 점수 -> 마스킹 -> softmax -> V 섞기).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ---- 조각 3: Q / K / V (글자 하나에서 질문 · 열쇠 · 값 세 줄 만들기) ----

class CausalSelfAttention(nn.Module):
    """attention 부품(앞 글자를 얼마나 참고할지 정하는 계산). 조각 3 의 Q / K / V 에 이어 점수 · 마스킹 · softmax · V 섞기까지 한다.

    Q (질문): 이 글자가 앞 글자들 중에서 무엇을 찾고 있는지
    K (열쇠): 이 글자가 남의 질문에 "나는 이런 글자야" 하고 내미는 표시
    V (값):   이 글자가 참고당할 때 건네줄 내용
    셋 다 같은 글자 줄 x 에서 만들지만, 바꿔 주는 기계(nn.Linear)가 셋으로 따로라서 결과가 서로 다르다.
    조각 4 의 비율표를 줄 한 덩어리로 한 장 만들던 것을, 조각 5 에서 줄을 n_head 갈래(head)로 쪼개 갈래마다 한 장씩 만든다.
    읽을 것은 4단계 문서의 4-1 ~ 4-5 (참고 비율 계산) · 5-1 ~ 5-4 (multi-head).
    """

    def __init__(self, config: ModelConfig):
        # nn.Module 을 물려받은 클래스의 첫 줄 (GPT 와 같은 이유 — 아래 기계들이 "부품 목록 장부" 에 적히게)
        super().__init__()

        # nn.Linear(들어오는 칸 수, 나가는 칸 수): 줄 하나를 받아 새 줄로 바꿔 주는 기계.
        # 안에 가중치(칸마다 곱하는 숫자)와 편향(더하는 숫자)이 있고, 처음엔 아무 값이다가 학습으로 고쳐진다.
        # 지금은 모양을 그대로 둔다: 입력 (B, T, n_embd) -> 출력 (B, T, n_embd). (칸을 쪼개는 건 조각 5 multi-head)

        # 질문을 만드는 기계.
        # 들어오는 것은 조각 1 · 2 를 거쳐 이미 줄이 된 x 라서 들어오는 칸 수가 n_embd 다 (vocab_size 는 "번호 -> 줄" 표의 줄 수일 뿐).
        # 나가는 칸 수도 지금은 같은 n_embd 로 둔다 — 조각 5 에서 이 줄을 4갈래로 쪼갤 때 128 이 필요하다.
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        # 결과: q_proj.weight.shape = torch.Size([128, 128]) (나가는 칸 x 들어오는 칸) · q_proj.bias.shape = torch.Size([128]) · 파라미터 16512 = 128 x 128 + 128

        # 열쇠를 만드는 기계. 질문 기계와 크기는 같지만 새로 만든다.
        # q_proj 를 다시 쓰면 가중치가 같아서 Q 와 K 가 똑같은 줄이 된다 (문서 3-3 ②).
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)

        # 값을 만드는 기계. 숫자 표(가중치)가 따로라서 학습하면 질문 · 열쇠와 다른 일을 하게 된다.
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)

        # ---- 조각 5: multi-head (문서 5-1 ~ 5-4) ----
        # 갈래 수와 갈래 하나의 칸 수. 128 칸 줄을 4 갈래로 쪼개면 갈래 하나는 32 칸 (128 // 4).
        # config 가 n_embd % n_head == 0 을 검사해 두었으니 나누어 떨어진다.
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # 갈래마다 따로 attention 한 결과를 이어 붙이면 갈래끼리는 아직 안 섞이고 나란히 놓여 있을 뿐이다.
        # nn.Linear 한 번을 더 통과시켜 갈래들의 결과를 서로 섞는다 (output projection, 문서 5-4 ④).
        # 파라미터가 128 x 128 + 128 = 16512 늘어난다.
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)

    def _split_heads(self, t):
        """줄을 n_head 갈래로 쪼개고 갈래 축을 글자 축 앞으로 보낸다. (B, T, C) -> (B, n_head, T, head_dim)"""
        B, T, C = t.shape

        # view 로 C 칸을 (갈래 수, 갈래 하나의 칸 수) 두 축으로 나눠 읽는다. (B, T, C) -> (B, T, n_head, head_dim)
        # 숫자는 그대로이고 묶는 방식만 바뀐다 (문서 5-2 ③).
        # 4, 32 를 직접 적지 않고 self 값을 쓰는 이유: n_head=8 로 바꿔도 4 x 32 = 128 이라 view 가 에러 없이 통과해
        # 갈래가 4개로만 쪼개진 채 틀린 값이 나온다 (내장 attention 과 비교해서 확인).
        t = t.view(B, T, self.n_head, self.head_dim)
        # 결과: (2, 5, 128) -> view 뒤 torch.Size([2, 5, 4, 32])

        # 글자 축 T(1번)와 갈래 축(2번)을 맞바꿔 갈래를 앞으로 보낸다. (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        # @ 와 softmax 는 맨 뒤 두 축만 보고 앞 축은 "묶음" 으로 취급한다. 갈래가 앞에 있어야 갈래마다 (T, head_dim) 표가 된다.
        # 결과: transpose 뒤 torch.Size([2, 4, 5, 32]) · is_contiguous() False (숫자는 안 옮기고 읽는 순서만 바뀜)
        return t.transpose(1, 2)

    def forward(self, x):
        """x: 조각 1 · 2 를 거친 글자 줄 묶음. shape (B, T, n_embd).
        반환: 앞 글자를 참고해 섞은 새 줄 묶음 y. shape (B, T, n_embd) — 입력과 같아서 뒤 부품이 그대로 이어받는다.
        """
        B, T, C = x.shape  # 문장 수 · 글자 수 · 줄의 칸 수

        # 조각 3: 같은 x 에서 질문 · 열쇠 · 값을 만든다. 셋 다 shape (B, T, C).
        # 조각 5: 만든 줄을 바로 n_head 갈래로 쪼갠다. 셋 다 shape (B, n_head, T, head_dim).
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # ---- ① 점수: 모든 글자 쌍의 질문-명찰이 얼마나 맞는지 (문서 4-1) ----
        # 질문 q 와 명찰 k 를 곱해서 더한다. @ 는 표 전체에 한 번에 해 주는 "곱해서 더하기".
        # @ 는 앞 표의 가로줄과 뒤 표의 세로줄을 짝짓는데 k 는 글자 하나가 가로줄이라, 뒤에서 두 번째 · 맨 뒤 축을 맞바꿔 뒤집는다.
        # 앞의 (B, n_head) 두 축은 묶음이라 문장마다 · 갈래마다 따로 계산된다 (문서 5-3 ②).
        # shape: (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T). 갈래마다 점수표 한 장
        scores = q @ k.transpose(-2, -1)
        # 결과: B=2, T=5, n_head=4 일 때 scores.shape = torch.Size([2, 4, 5, 5])

        # ---- 크기 줄이기: 칸이 많으면 점수가 커져 softmax 가 한 글자에 100% 를 몰아 주므로 sqrt(칸 수)로 나눈다 (문서 4-2) ----
        # 곱해서 더하는 줄이 이제 갈래 하나라서, 조각 4 의 sqrt(C) 대신 갈래 하나의 칸 수 sqrt(head_dim) 을 쓴다 (문서 5-3 ③).
        scores = scores / math.sqrt(self.head_dim)

        # ---- ② 마스킹: 뒤 글자를 못 보게 가린다 (문서 4-3) ----
        # 볼 수 있는 칸 = 1 · 못 보는 칸 = 0 인 지도. tril 은 왼쪽 아래 삼각형만 남기고 나머지를 0 으로 만든다.
        # device=x.device: 지도를 입력과 같은 장치(GPU/CPU)에 만든다. 다르면 점수표와 계산이 안 된다.
        # 지도는 (T, T) 한 장이다. 점수표가 (B, n_head, T, T) 여도 앞 두 축으로 자동 복사되어 모든 갈래에 같은 지도가 적용된다.
        mask = torch.tril(torch.ones(T, T, device=x.device))
        # 지도가 0 인 칸을 마이너스 무한대로 덮는다. -inf 는 softmax 를 거치면 정확히 0% 가 된다 (문서 4-4 ③).
        # 0 점으로 덮으면 "조금은 참고" 가 되어 0% 가 아니라서 -inf 를 쓴다.
        scores = scores.masked_fill(mask == 0, float("-inf"))

        # ---- ③ softmax: 줄마다 점수를 합 1 인 %로 (문서 4-4) ----
        # dim=-1: 맨 뒤 축 방향 = 한 줄 안의 칸끼리 합을 1 로 맞춘다. 점수표의 한 줄이 한 글자의 참고 비율이라서다.
        att = F.softmax(scores, dim=-1)
        # att: (B, n_head, T, T) — 줄마다 합 1, 가려진 칸은 0
        # 결과: att.shape = torch.Size([2, 4, 5, 5]) · 줄 합 1.0 (torch.manual_seed(0) · x = randn(2, 5, 128)) · 갈래마다 비율표가 다르다
        #       첫 문장 갈래 0 마지막 줄 [0.11, 0.16, 0.23, 0.23, 0.28] · 갈래 1 마지막 줄 [0.27, 0.11, 0.19, 0.13, 0.30]

        # ---- ④ 가져오기: % 대로 노트(v)를 섞어 새 줄을 만든다 (문서 4-5) ----
        # 비율표와 노트를 곱해서 더한다. 앞에 비율표, 뒤에 노트.
        # (B, n_head, T, T) @ (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v
        # 결과: y.shape = torch.Size([2, 4, 5, 32]) — 아직 갈래가 나뉜 채

        # ---- ⑤ 합치기: 갈래들을 한 줄로 이어 붙이고 한 번 더 섞는다 (문서 5-4) ----
        # 갈래 축을 글자 축 뒤로 되돌린 (B, T, n_head, head_dim) 을 한 줄 C 칸으로 이어 읽는다.
        # contiguous: transpose 는 읽는 순서만 바꾸므로, view 전에 숫자를 새 순서대로 실제로 정리한다. 안 하면 view 가 에러 (문서 5-4 ③).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 결과: 되돌린 뒤 torch.Size([2, 5, 4, 32]) · is_contiguous() False -> contiguous().view 뒤 torch.Size([2, 5, 128])

        # 갈래끼리 섞는다. shape (B, T, C) 그대로.
        y = self.o_proj(y)
        # 결과: y.shape = torch.Size([2, 5, 128]) — 입력 x 와 같은 모양. F.scaled_dot_product_attention 을 갈래별로 쓴 것과 allclose True
        #       (n_head = 1 · 2 · 4 · 8 모두). 마지막 글자만 바꿔도 앞 4글자 결과 그대로 (마스킹이 갈래마다 지켜진다)

        return y


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

    # 조각 3 확인: 위치까지 섞은 out 을 attention 부품에 넣어 Q / K / V 를 만든다
    attn = CausalSelfAttention(config)
    y = attn(out)
    print(y.shape)  # 기대: torch.Size([2, 5, 128]) (입력과 같은 모양)
    # 기대: 66048 = 조각 3 의 49536 + o_proj 16512 (128 x 128 + 128). 갈래로 쪼개기 · 점수 · softmax 는 학습할 숫자가 없다
    print(sum(p.numel() for p in attn.parameters()))

    # 마스킹 확인: 마지막 글자만 바꿔도 앞 글자들의 새 줄은 그대로여야 한다 (뒤 글자를 못 보니까)
    out2 = out.clone()
    out2[:, -1] += 1.0
    y2 = attn(out2)
    print(torch.allclose(y[:, :-1], y2[:, :-1]))  # 기대: True
    print(torch.allclose(y[:, -1], y2[:, -1]))  # 기대: False (마지막 글자 자신은 바뀐다)

# ---- 직접 실행 결과 (uv run python -m llm.model) ----
# torch.Size([2, 5])
# torch.Size([2, 5, 128])
# 25216
# torch.Size([2, 5, 128])
# 66048
# True
# False
