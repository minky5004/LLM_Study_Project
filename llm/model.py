"""트랜스포머 아키텍처: 임베딩 -> attention -> FFN -> 조립.

GPT-2식으로 먼저 완성하고, 돌아가면 부품을 하나씩 현대식으로 교체한다 (4단계 문서 참고).
지금은 조각 7까지: 토큰 임베딩 + 위치 임베딩 (몇 번째 글자인지 알려 주기) + multi-head attention + 블록 (FFN · 잔차 · LayerNorm) + 블록 쌓기 · 출력층.
8단계 조각 3: dropout 을 실제로 켠다 — 임베딩 합 뒤 · 블록 안 attention · FFN 출력 뒤 (8단계 문서 3-3).
10단계 조각 3: LayerNorm 13곳을 직접 짠 RMSNorm 으로 교체 (10단계 문서 3-1 ~ 3-4).
10단계 조각 4: 위치 임베딩 표(wpe) 삭제 · 위치는 attention 안에서 q · k 를 돌려 알려 준다 — RoPE (10단계 문서 4-1 ~ 4-5).
10단계 조각 5: FFN 의 ReLU 를 SwiGLU 로 — 문 줄(SiLU) × 내용 줄 · 펼치는 칸 4배 → 8/3배로 숫자 수 맞춤 (10단계 문서 5-1 ~ 5-4).
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
        # 10단계 조각 4: 이 표를 지웠다 — 위치는 attention 안에서 q · k 를 돌려 알려 준다 (RoPE · 10단계 문서 4-1).
        # 결과 (지우기 전): self.wpe.weight.shape = torch.Size([128, 128]) — 줄 128개(자리) × 칸 128개 · 학습 크기(256칸)에선 32,768개

        # 8단계 조각 3: 글자 줄 + 자리 줄을 합친 직후에 쓸 dropout (8단계 문서 3-3).
        # 학습 모드일 때만 config.dropout 비율만큼 숫자를 무작위로 0 으로 지운다. 학습할 숫자가 없어 파라미터 수는 그대로다.
        self.drop = nn.Dropout(config.dropout)

        # ---- 조각 7: 블록 쌓기 · 마지막 LayerNorm · 출력층 ----

        # 블록 n_layer 개를 담는 부품 리스트. 블록은 입력과 출력 모양이 같아서 줄줄이 이어 붙일 수 있다 (문서 7-1).
        # nn.ModuleList 로 감싸는 이유: 일반 리스트에 넣으면 torch 가 그 안의 학습 숫자를 장부에 못 올린다.
        # 같은 블록 하나를 n_layer 번 복사하면 안 된다 — 층마다 자기만의 숫자를 따로 학습해야 해서 Block(config) 를 n_layer 번 "새로" 만든다.
        # [Block(config) for _ in range(...)] — 반복문을 한 줄로 써서 리스트를 만드는 문법(컴프리헨션). `_` 는 반복 번호를 쓰지 않는다는 표시다.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # 결과: len(self.blocks) = 4 · type = ModuleList · blocks[0] 과 blocks[1] 의 FFN 가중치가 서로 다르다 (실행 확인 — 층마다 따로 학습)

        # 마지막 LayerNorm (문서 7-2). 블록은 부품 "앞" 에서 줄을 맞추기 때문에 마지막 블록의 출력은 맞춰지지 않은 채 나온다.
        # 출력층에 넣기 전에 한 번 더 맞춘다. 맞출 줄 한 개의 칸 수(맨 뒤 축)가 n_embd 라서 괄호에 n_embd 를 적는다.
        # 10단계 조각 3: LayerNorm 대신 RMSNorm — 이름 ln_f 는 그대로 둔다 (부르는 쪽 코드를 안 바꾸려고)
        self.ln_f = RMSNorm(config.n_embd)
        # 결과 (LayerNorm 때): 파라미터 256 = gamma 128 + beta 128 · 장부에 ln_f.weight · ln_f.bias 로 올라간다 (실행 확인)

        # 출력층 (문서 7-3). 글자 줄(n_embd 칸)을 "글자 종류마다 점수 한 칸" 짜리 줄로 바꾼다.
        # 들어오는 칸 수는 글자 줄의 n_embd · 나가는 칸 수는 글자 종류 수 vocab_size (칸 하나 = 글자 하나의 점수).
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        # 결과: lm_head.weight.shape = torch.Size([69, 128]) · 파라미터 8901 = 128 x 69 + 69

    def forward(self, idx):
        """forward = 데이터가 실제로 흘러가는 곳. 모델을 부를 때마다 실행된다.

        idx: 글자 번호 묶음. shape (B, T) — B = 문장 몇 개를 한꺼번에, T = 문장 하나의 글자 몇 개.
        반환: 자리마다 "다음 글자가 각 글자일 점수" 가 붙은 텐서(logits). shape (B, T, vocab_size).
        """
        # 번호 묶음을 표에 넣으면, 번호마다 그 번호의 줄이 꺼내져 나온다 (계산이 아니라 표 찾기).
        # 번호 하나가 숫자 n_embd 개로 바뀌므로 shape 맨 뒤에 n_embd 가 붙는다: (B, T) -> (B, T, n_embd)
        tok_emb = self.wte(idx)
        # 결과: idx (2, 5) -> tok_emb (2, 5, 128). 같은 번호는 같은 줄을 받는다 (번호 [0, 1, 0] 의 첫째 · 셋째 줄이 똑같음 — 실행 확인)

        # 10단계 조각 4: 자리 줄을 더하던 단계(자리 번호 만들기 · wpe 에서 꺼내기 · tok_emb + pos_emb)를 지웠다.
        # 이제 글자 줄에는 위치가 없다 — 번호 [0, 1, 0] 의 첫째 · 셋째 줄이 다시 같다. 위치는 attention 이 q · k 를 돌려서 알려 준다.
        x = tok_emb

        # 글자 줄에 dropout. 모양은 그대로 (B, T, n_embd) — 숫자 일부가 0 이 될 뿐이다. 점검 모드(eval)면 아무것도 안 한다.
        x = self.drop(x)

        # 블록을 순서대로 통과시킨다. 앞 블록의 출력이 다음 블록의 입력이다. 블록은 모양을 안 바꾸므로 x 는 계속 (B, T, n_embd).
        for block in self.blocks:
            x = block(x)
        # 결과: 블록 4개를 지나도 x 는 계속 (2, 5, 128) — 모양이 안 바뀌어서 이어 붙일 수 있다

        # 마지막 LayerNorm 으로 줄을 맞춘 뒤 출력층에 넣는다. (B, T, n_embd) -> (B, T, vocab_size)
        # logits = softmax 를 거치기 전의 날점수. 확률로 바꾸는 일은 학습 · 생성 단계에서 한다.
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # 결과: logits.shape = torch.Size([2, 5, 69]) — 자리마다 글자 종류별 점수 69개
        #       한 줄의 합이 1 이 아니다 (-1.349 — 날점수의 증거) · 전체 파라미터 827461
        #       마지막 글자만 바꿔도 앞 4글자의 logits 는 그대로 True (모델 전체에서도 마스킹이 지켜진다)
        return logits


# ---- 10단계 조각 4: RoPE — q · k 를 자리만큼 돌려 위치 알려 주기 (10단계 문서 4-1 ~ 4-5) ----

def rope_tables(head_dim, block_size, base=10000.0):
    """자리마다 · 짝마다 돌릴 각도의 cos 표 · sin 표를 만든다. 둘 다 shape (block_size, head_dim // 2).

    학습하는 숫자가 아니라 자리와 속도로 정해지는 고정값이라, 모델을 만들 때 한 번만 부른다 (문서 4-5 ②).
    """
    # 짝마다 도는 속도 (문서 4-4 ③). 짝 i 의 속도 = 1 / base^(2i / head_dim).
    # torch.arange(0, head_dim, 2) = [0, 2, 4, ..., head_dim-2] 가 곧 2i 다. 짝 수만큼 head_dim // 2 개.
    # 첫 짝은 1 (한 칸에 1 라디안 — 빠른 초침) · 뒤로 갈수록 느려져 마지막 짝은 거의 안 돈다 (시침).
    speed = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # 결과: head_dim = 32 일 때 speed.shape = (16,) · speed[0] = 1.0 · speed[1] = 0.5623 · speed[-1] = 0.000178

    # 자리 번호 [0, 1, ..., block_size-1]. 모델이 볼 수 있는 최대 길이만큼 미리 만든다 (입력 길이 T 는 forward 에서 잘라 쓴다).
    pos = torch.arange(block_size).float()

    # 각도 = 자리 × 속도 (문서 4-4 ④). 모든 (자리, 짝) 쌍마다 하나씩 → shape (block_size, head_dim // 2).
    # 그냥 pos * speed 는 길이가 달라 에러. pos 에 [:, None] 을 붙여 세로 막대 (block_size, 1) 로 세우면
    # broadcasting(모양 자동 맞추기)이 가로 막대 speed (head_dim // 2,) 와 곱해 표로 펼쳐 준다 — 줄 = 자리 · 칸 = 짝.
    # speed 쪽을 세우면 (head_dim // 2, block_size) 로 뒤집혀 apply_rope 의 cos[:T] 가 자리 대신 짝을 잘라 에러가 난다.
    angles = pos[:,None] * speed
    # 결과: head_dim = 32 · block_size = 128 일 때 angles.shape = (128, 16) · 자리 1 줄 = speed 그대로 · 자리 0 줄은 전부 0
    #       speed[:, None] 으로 세우면 (16, 128) → RuntimeError (size of tensor a (16) must match the size of tensor b (128))

    return angles.cos(), angles.sin()


def apply_rope(t, cos, sin):
    """q 나 k 를 자리만큼 돌린다. t: (B, n_head, T, head_dim) -> 같은 모양.

    칸을 두 개씩 짝지어 (앞 칸 a, 뒤 칸 b) 를 평면 화살표로 보고, 짝마다 각도만큼 돌린다 (문서 4-3).
    """
    T = t.shape[-2]  # 이번 입력의 글자 수

    # 각도표는 block_size 자리만큼 만들어 두었으니 앞 T 자리만 쓴다. (block_size, head_dim // 2) -> (T, head_dim // 2)
    # (T, head_dim // 2) 는 맨 뒤 두 축이 t 의 (T, head_dim // 2) 와 같아서, 앞의 (B, n_head) 로 자동 복사되어 곱해진다.
    cos, sin = cos[:T], sin[:T]

    # 짝의 앞 칸 묶음 a · 뒤 칸 묶음 b 로 떼어 낸다 (문서 4-3 ④). 둘 다 (B, n_head, T, head_dim // 2).
    # `...` 은 "앞 축은 전부 그대로" · 맨 뒤 축만 고른다. 고르는 문법은 "시작:끝:간격" (끝을 비우면 끝까지).
    # 짝의 앞 칸 = 0, 2, 4, ... 번 칸 (0::2) · 뒤 칸 = 1, 3, 5, ... 번 칸 (1::2)
    a = t[..., 0::2]
    b = t[..., 1::2]

    # 짝마다 각도만큼 돌리는 공식 (문서 4-3 ②). 칸마다 따로 계산되어 모양은 그대로 (B, n_head, T, head_dim // 2).
    # 새 가로(앞 칸) = a cos − b sin · 새 세로(뒤 칸) = a sin + b cos.
    # 가운데 부호가 빼기여야 돌리기다 — 더하기면 화살표 길이가 바뀌고 "거리만 남는다" 가 깨진다.
    # (1, 0) 을 90도 돌리는 예로는 b = 0 이라 부호 실수가 안 보인다. (0, 1) → (-1, 0) 으로 확인한다.
    new_a = a * cos - b * sin
    new_b = a * sin + b * cos
    # 결과: 돌린 뒤 줄 길이가 돌리기 전과 allclose True · (자리 3, 자리 1) 과 (자리 7, 자리 5) 의 q·k 점수가 같다 (아래 확인 출력)

    # 돌린 두 묶음을 원래 칸 자리로 되돌린다 — 짝수 번 칸에 new_a · 홀수 번 칸에 new_b.
    # stack(dim=-1) 로 짝을 맨 뒤 축에 나란히 세우고 (…, head_dim // 2, 2) · flatten(-2) 로 두 축을 한 줄로 이어 읽는다 (…, head_dim).
    # 결과 순서: [new_a[0], new_b[0], new_a[1], new_b[1], ...] = 원래의 [0번, 1번, 2번, 3번, ...] 칸 자리
    return torch.stack([new_a, new_b], dim=-1).flatten(-2)


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

        # 10단계 조각 4: RoPE 각도표 (문서 4-5). 갈래 하나의 칸 수(head_dim)를 짝지어 돌리므로 head_dim 으로 만든다.
        # register_buffer: 학습은 안 하지만 model.to("cuda") 때 같이 GPU 로 옮겨지는 칸. 그냥 self.rope_cos = ... 면 CPU 에 남아 에러.
        # persistent=False: 모델 파일(state_dict)에는 저장하지 않는다 — 언제든 같은 값으로 다시 계산되니까.
        cos, sin = rope_tables(self.head_dim, config.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

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

        # 10단계 조각 4: 점수를 만드는 q · k 만 자리만큼 돌린다 (문서 4-5 ①). v 는 점수가 정해진 뒤 섞일 내용이라 그대로.
        # 모양은 그대로 (B, n_head, T, head_dim) — 돌리기는 길이를 안 바꾸고 각도만 바꾼다 (문서 4-2 ④).
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

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


# ---- 조각 6: FFN · 잔차 · LayerNorm 으로 블록 하나 완성 (문서 6-1 ~ 6-5) ----

def silu(x):
    """SiLU (Swish): x × sigmoid(x). 큰 양수는 거의 그대로 · 큰 음수는 거의 0 · 0 근처는 부드럽게 (10단계 문서 5-2).

    sigmoid 는 숫자를 0 ~ 1 사이 비율로 바꾸는 함수 — torch.sigmoid 로 부른다. 칸마다 따로 계산되어 모양은 그대로.
    """
    # x 에 "x 의 sigmoid" 를 칸마다 곱한다
    return x * torch.sigmoid(x)
    # 결과: silu([2, -2, 0, -1.28, 5, -5]) = [1.7616, -0.2384, 0.0000, -0.2785, 4.9665, -0.0335] — -1.28 근처가 가장 낮은 값 -0.28


class FeedForward(nn.Module):
    """FFN: 글자마다 따로 줄을 가공하는 작은 신경망. shape (B, T, n_embd) -> (B, T, n_embd).

    attention 이 글자끼리 섞어 내용을 모아 온 뒤, 그 내용을 글자 하나 안에서 가공한다 (문서 6-1).
    글자끼리는 섞지 않는다 — 첫 글자만 바꿔도 다른 글자의 결과는 그대로다.
    10단계 조각 5: GPT-2 의 Linear(펼치기) -> ReLU -> Linear(접기) 를 SwiGLU 로 바꿨다 (10단계 문서 5-1 ~ 5-4).
    같은 입력에서 문 줄(gate)과 내용 줄(up)을 펼쳐 만들고, 문에 SiLU 를 씌워 칸마다 곱한 뒤 접는다(down).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # 펼치는 칸 수. GPT-2 는 4배(Linear 2개)였다. SwiGLU 는 Linear 가 3개라 4배 그대로면 숫자가 1.5배가 된다.
        # 숫자 수를 맞춰야 "구조 덕인지 숫자가 늘어난 덕인지" 가릴 수 있어서 4배의 2/3 = 8/3배로 줄인다 (문서 5-3).
        # 4 * n_embd 의 2/3 — 칸 수는 정수여야 해서 int(...) 로 감싼다 (소수점 아래는 버림)
        hidden = int(4 * config.n_embd * 2 / 3)
        # 결과: n_embd=128 이면 hidden = 341 (341.33 버림) · 256 이면 682 · GPT-2 때는 512 · 1024

        # 문 줄을 만드는 기계와 내용 줄을 만드는 기계. 둘 다 n_embd -> hidden 으로 같은 모양이지만 숫자가 따로라 다른 줄이 나온다
        # (조각 3 에서 q_proj · k_proj 를 따로 만든 것과 같은 이유).  shape (B, T, n_embd) -> (B, T, hidden)
        self.gate = nn.Linear(config.n_embd, hidden)
        self.up = nn.Linear(config.n_embd, hidden)
        # 접는 기계: hidden -> n_embd. 입력과 같은 칸 수로 돌아와야 잔차에서 더할 수 있다 (문서 6-3 ④). (GPT-2 때 이름 fc2)
        self.down = nn.Linear(hidden, config.n_embd)
        # 결과 (GPT-2 때): FFN 파라미터 131712 = fc1 (128 x 512 + 512) + fc2 (512 x 128 + 128)

    def forward(self, x):
        # 문 줄에 SiLU 를 씌워 "칸마다 통과량" 으로 만들고, 내용 줄과 칸마다 곱한다 (문서 5-1 ③ · 5-2).
        # 둘 다 (B, T, hidden) 이라 * 는 같은 자리 칸끼리 곱한다. shape (B, T, n_embd) -> (B, T, hidden)
        x = silu(self.gate(x)) * self.up(x)

        # 접는다. shape (B, T, hidden) -> (B, T, n_embd)
        x = self.down(x)
        return x


# ---- 10단계 조각 3: RMSNorm — 크기만 맞추는 LayerNorm (10단계 문서 3-1 ~ 3-4) ----

class RMSNorm(nn.Module):
    """글자마다 줄의 크기(제곱 평균의 제곱근)를 1 근처로 맞추고 칸마다 gamma 를 곱한다. shape 그대로 (..., dim) -> (..., dim).

    LayerNorm 과 다른 점: 평균 빼기 · beta 더하기가 없다 (문서 3-1). 그래서 숫자가 dim 개(gamma 만) — LayerNorm 은 2 x dim.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        # 0 으로 나누지 않게 제곱 평균에 더할 아주 작은 수 (문서 3-2 ④). nn.RMSNorm 과 비교할 때 같은 값을 넘긴다.
        self.eps = eps
        # gamma: 칸마다 곱할 학습 숫자. nn.Parameter 로 감싸야 장부에 올라가 학습된다 (이름 weight 는 nn.RMSNorm 과 같게).
        # 처음엔 "곱해도 그대로" 가 되게 1 로 채운다 (문서 3-3) — LayerNorm 의 gamma 와 같은 출발
        self.weight = nn.Parameter(torch.ones(dim))
        # 결과: RMSNorm(128) 파라미터 128 (gamma 만 · LayerNorm 은 gamma + beta 256)

    def forward(self, x):
        # 칸마다 제곱 → 맨 뒤 축(칸 dim 개)을 따라 평균. shape (B, T, dim) -> (B, T, 1)
        # keepdim=True: 평균을 낸 축을 크기 1 로 남긴다 — 남겨야 (B, T, dim) 을 (B, T, 1) 로 나눌 때 칸마다 같은 값으로 나눠진다
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        # 결과: (2, 5, 128) 을 넣으면 ms.shape = (2, 5, 1) · keepdim=False 면 (2, 5) 라 다음 줄 나누기가
        #       RuntimeError (size of tensor a (128) must match the size of tensor b (5)) — 맨 뒤 축끼리 맞추려 해서
        # 제곱 평균에 eps 를 더하고 제곱근 → 그 값으로 나눈다 (문서 3-2). shape (B, T, dim) / (B, T, 1) -> (B, T, dim)
        # 숫자 1e-6 을 박지 않고 self.eps — RMSNorm(dim, eps=1e-5) 처럼 만들 때 넘긴 값이 쓰이게
        x = x / torch.sqrt(ms + self.eps)
        # 칸마다 gamma 를 곱한다. (B, T, dim) * (dim,) — 맨 뒤 축끼리 맞춰 모든 글자에 같은 gamma
        return x * self.weight
        # 결과: RMSNorm(2)([[3., -4.]]) = [[0.8485, -1.1314]] — RMS 3.54 로 나눔 · 부호와 비율 그대로 (문서 3-2 ③ 의 예)


class Block(nn.Module):
    """트랜스포머 블록 하나: attention 과 FFN 을 LayerNorm · 잔차로 묶는다. shape (B, T, n_embd) -> (B, T, n_embd).

    입력과 출력 모양이 같아서 이 블록을 그대로 여러 겹 쌓을 수 있다 (조각 7).
    순서는 GPT-2 의 pre-norm: 부품 앞에서 줄을 맞추고(LayerNorm) -> 부품 -> 입력을 더한다 (문서 6-5).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # LayerNorm: 글자마다 줄을 평균 0 · 퍼짐 1 근처로 맞춘다 (문서 6-4). 괄호에는 맞출 줄의 칸 수(맨 뒤 축)를 적는다.
        # attention 앞 · FFN 앞에 따로 둔다 — 안에 학습되는 숫자(gamma · beta)가 각자 있어서 둘은 다른 층이다.
        # 글자 하나의 줄이 n_embd 칸이라 두 곳 모두 n_embd 를 넣는다. 안에 칸마다 gamma · beta 가 있어 파라미터가 128 x 2 = 256 이다.
        # 10단계 조각 3: 두 곳 모두 RMSNorm 으로 — 이름 ln1 · ln2 는 그대로 (문서 3-4 ③)
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.ffn = FeedForward(config)
        # 결과: ln1 파라미터 128 · FFN 131712 · attention 66048 · ln2 128 -> 블록 하나 198016 (LayerNorm 때 256 · 256 -> 198272)

        # 8단계 조각 3: attention · FFN 의 출력을 잔차로 더하기 "전에" 지울 dropout (8단계 문서 3-3).
        # 더하는 길(x 그대로)은 지우지 않는다 — 원래 정보는 그대로 흐르고, 부품이 새로 만든 몫만 일부 지운다.
        # 학습할 숫자가 없어 하나를 두 곳에서 같이 써도 된다.
        # 지울 비율은 1단계부터 설정 상자에 있던 config.dropout (7단계까지는 이름만 있고 쓰이지 않았다).
        self.drop = nn.Dropout(config.dropout)
        # 결과: 블록 하나 파라미터 198272 그대로 (dropout 은 학습할 숫자가 없다) · 모델 전체 827461 그대로

    def forward(self, x):
        # 잔차 연결: 부품의 출력에 입력을 그대로 더한다 (문서 6-3). 부품은 "바꿀 만큼" 만 만들면 된다.
        # 더하는 길에는 LayerNorm 이 없어서 입력의 원래 정보가 그대로 흐른다.
        # x + 부품(LayerNorm(x)) 가 한 묶음이고 attention 으로 한 번 · FFN 으로 한 번 한다 (문서 6-5).
        # 더하는 쪽 x 는 LayerNorm 을 안 거친 입력 그대로다. 더하려면 모양이 같아야 해서 (B, T, n_embd) + (B, T, n_embd).
        # 8단계 조각 3: 부품의 출력을 더하기 전에 dropout 을 한 번 거친다 — x + 지우기(부품(LayerNorm(x))).
        # 두 줄 모두 부품의 출력을 __init__ 에서 만든 self.drop 으로 감싼다.
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.ffn(self.ln2(x)))
        # 결과 (8단계 조각 3): 6층 · 256칸 · dropout 0.2 로 5000번 학습 → 검증 손실 최저 1.6435 (dropout 0 이면 같은 크기에서 1.72 뒤 외우기)
        # 결과: 블록 출력 shape = torch.Size([2, 5, 128]) — 입력과 같은 모양
        #       attention 의 o_proj · FFN 의 접기(GPT-2 때 fc2 · 지금 down)를 0 으로 만들면 블록이 입력을 그대로 돌려준다 (allclose True — 잔차의 증거)
        #       마지막 글자만 바꿔도 앞 4글자의 출력은 그대로 True (블록 전체에서도 마스킹이 지켜진다)
        #       ModelConfig(n_embd=64, n_head=2) · (n_embd=256, n_head=8) 도 같은 코드로 모양이 맞게 돈다
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

    # 조각 7 부터 model(idx) 는 글자별 점수(logits)를 낸다. 조각 1~6 확인은 "임베딩만 거친 줄" 이 필요해서 표를 직접 불러 만든다.
    # 10단계 조각 4: 자리표 wpe 를 지워 글자 표만 거친다
    out = model.wte(idx)
    print(idx.shape)   # 기대: torch.Size([2, 5])
    print(out.shape)   # 기대: torch.Size([2, 5, 128])
    # 기대: 8832 = 글자 표 (69 x 128) · 자리표가 있을 때 25216 (+ 128 x 128)
    print(model.wte.weight.numel())

    # 조각 7 확인: 모델 전체. 번호 (B, T) 가 들어가 글자 종류마다 점수 한 칸씩 (B, T, vocab_size) 이 나온다
    logits = model(idx)
    print(logits.shape)  # 기대: torch.Size([2, 5, 69])
    # 기대: 810093 = 글자 표 8832 + 블록 198058 x 4 + ln_f 128 + 출력층 8901 (128 x 69 + 69) · ReLU FFN 때 809925 (+168 = 42 x 4) · 자리표 있을 때 826309 · LayerNorm 때 827461
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

    # 조각 6 확인: 블록 하나. 입력과 같은 모양이어야 쌓을 수 있다
    block = Block(config)
    z = block(out)
    print(z.shape)  # 기대: torch.Size([2, 5, 128])
    # 기대: 198058 = attention 66048 + RMSNorm 128 x 2 + SwiGLU FFN 131754 · ReLU FFN 때 198016 (FFN 131712) · LayerNorm 때 198272
    print(sum(p.numel() for p in block.parameters()))

    # 잔차 확인: 부품의 마지막 Linear(o_proj · down)를 0 으로 만들면 부품 출력이 0 이라 블록은 입력을 그대로 돌려줘야 한다
    zero_block = Block(config)
    with torch.no_grad():
        for lin in (zero_block.attn.o_proj, zero_block.ffn.down):
            lin.weight.zero_()
            lin.bias.zero_()
    print(torch.allclose(zero_block(out), out))  # 기대: True

    # FFN 확인: 글자끼리 섞지 않는다 — 마지막 글자만 바꿔도 앞 글자들의 결과는 그대로
    ffn = FeedForward(config)
    print(torch.allclose(ffn(out)[:, :-1], ffn(out2)[:, :-1]))  # 기대: True

    # 10단계 조각 5 확인: SwiGLU (문서 5-2 ~ 5-4)
    sample = torch.tensor([2.0, -2.0, 0.0, -1.28, 5.0, -5.0])
    print(silu(sample))  # 기대: [1.7616, -0.2384, 0.0000, -0.2785, 4.9665, -0.0335] — 큰 양수는 그대로 · 큰 음수는 0 근처 · 작은 음수는 조금 통과
    print(torch.allclose(silu(out), F.silu(out)))  # 기대: True — 직접 짠 silu == 내장 F.silu
    print(ffn.gate.out_features)  # 기대: 341 = int(4 x 128 x 2/3) · GPT-2 때 512
    # 기대: 131754 = 문 · 내용 (128 x 341 + 341) x 2 + 접기 (341 x 128 + 128) · GPT-2 때 131712 (+42 — 거의 같음)
    print(sum(p.numel() for p in ffn.parameters()))

    # 10단계 조각 3 확인: 직접 짠 RMSNorm 이 내장 nn.RMSNorm 과 같은 값을 내는가 (문서 3-4)
    mine = RMSNorm(config.n_embd)
    ref = nn.RMSNorm(config.n_embd, eps=1e-6)  # eps 를 같게 — 안 주면 nn.RMSNorm 은 다른 기본값을 쓴다
    with torch.no_grad():
        mine.weight.copy_(torch.randn(config.n_embd))  # gamma 가 1 이 아니어도 같은지 보려고 무작위로
        ref.weight.copy_(mine.weight)  # 둘에 같은 gamma
    print(torch.allclose(mine(out), ref(out)))  # 기대: True
    print(sum(p.numel() for p in mine.parameters()))  # 기대: 128 (gamma 만 — LayerNorm 은 256)
    # 크기 맞추기 확인: gamma 가 1 이면 글자마다 줄의 제곱 평균이 1 근처
    print(RMSNorm(config.n_embd)(out).pow(2).mean(dim=-1)[0])  # 기대: 1 근처 5개

    # 10단계 조각 4 확인: RoPE (문서 4-2)
    head_dim = config.n_embd // config.n_head
    cos, sin = rope_tables(head_dim, config.block_size)
    print(cos.shape)  # 기대: torch.Size([128, 16]) — 자리 128 × 짝 16
    # 같은 q 줄 · 같은 k 줄을 10 자리 모두에 놓고 돌린다. (1, 1, 10, head_dim) = 문장 1 · 갈래 1 · 글자 10
    q_line, k_line = torch.randn(head_dim), torch.randn(head_dim)
    qr = apply_rope(q_line.expand(1, 1, 10, head_dim), cos, sin)[0, 0]  # (10, head_dim) — 자리마다 돌린 q
    kr = apply_rope(k_line.expand(1, 1, 10, head_dim), cos, sin)[0, 0]
    # 돌려도 길이는 그대로 (문서 4-2 ④)
    print(torch.allclose(qr.norm(dim=-1), q_line.norm().expand(10)))  # 기대: True
    # 거리만 남는다: (자리 3, 자리 1) 과 (자리 7, 자리 5) 는 둘 다 거리 2 → 점수가 같다
    print(torch.allclose(qr[3] @ kr[1], qr[7] @ kr[5], atol=1e-5))  # 기대: True
    # 거리가 다르면 점수도 다르다: (3, 1) 거리 2 vs (3, 2) 거리 1
    print(torch.allclose(qr[3] @ kr[1], qr[3] @ kr[2], atol=1e-5))  # 기대: False

# ---- 직접 실행 결과 (uv run python -m llm.model) ----
# torch.Size([2, 5])
# torch.Size([2, 5, 128])
# 8832                       <- 10단계 조각 4 (RoPE) · 자리표 있을 때 25216
# torch.Size([2, 5, 69])
# 809925                     <- 10단계 조각 4 (RoPE · 자리표 -16384) · RMSNorm 만 826309 · LayerNorm 때 827461
# torch.Size([2, 5, 128])
# 66048
# True
# False
# torch.Size([2, 5, 128])
# 198016                     <- LayerNorm 때 198272
# True
# True
# True                       <- 직접 짠 RMSNorm == nn.RMSNorm (allclose · 무작위 gamma)
# 128
# tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000], grad_fn=<SelectBackward0>)  <- 글자 5개 모두 제곱 평균 1
# torch.Size([128, 16])      <- RoPE 각도표: 자리 128 × 짝 16
# True                       <- 돌려도 줄 길이 그대로
# True                       <- (3, 1) 과 (7, 5) — 거리 2 끼리 점수 같음
# False                      <- (3, 1) 과 (3, 2) — 거리 2 와 1 은 점수 다름
# 학습 크기(6층 · 256칸 · BPE 1024) 파라미터 5,293,824 — LayerNorm 때 5,297,152 (-3,328 = 256 x 13곳 · -0.06%)
