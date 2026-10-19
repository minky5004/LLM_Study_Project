"""추론·텍스트 생성.

지금은 조각 4까지: 저장한 모델 불러오기 · 시작 글을 번호 텐서로 (문서 1-1 ~ 1-2) · 다음 글자 하나 뽑기 (문서 2-1 ~ 2-2) · 이어 쓰기 (문서 3-1 ~ 3-2) · 대화 (문서 4-1 ~ 4-2).
조각 4 가 마지막이다: 사용자 말을 시작 글로 만들고 · 답 한 줄만 잘라 · 터미널 대화로 묶는다.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from llm.config import ModelConfig
from llm.model import GPT

# 주의: tokenizer 를 import 하면 그 모듈 최상위가 실행되어 train.txt 를 읽는다. 프로젝트 루트에서 `-m` 으로 실행해야 경로가 맞는다.
from llm.tokenizer import decode, encode, vocab_size

# 6단계 train.py 를 import 하지 않는다 — import 만 해도 모델을 만들고 한 걸음 학습하는 줄이 최상위에 있어서다.
# 필요한 것(장치 · 설정 · 파일 경로)은 여기서 다시 정한다.


# ---- 조각 1: 저장한 모델 불러오기 · 시작 글을 번호 텐서로 ----

# 학습 때와 같은 장치를 고른다 (6단계 문서 1-1).
device = "cuda" if torch.cuda.is_available() else "cpu"

# train.py 가 저장한 파일. 8단계 조각 3 부터 {"config": 모델 설정, "model": 학습한 숫자} 사전이 들어 있다.
checkpoint_path = Path("checkpoints/model.pt")
ckpt = torch.load(checkpoint_path, map_location=device)

# 학습 때와 똑같은 설정으로 빈 모델을 만든다. 구조가 다르면 저장한 숫자가 들어갈 칸이 안 맞아 에러가 난다 (6단계 문서 5-2 ③).
# 설정은 파일에 적힌 것을 쓴다 — 학습 때 크기(6층 · 256칸)를 바꿔도 여기는 고칠 것이 없다 (8단계 문서 3-5).
# ModelConfig(**사전): 사전의 이름표를 인자 이름으로 풀어 넣는다.
model_config = ModelConfig(**ckpt["config"])
model = GPT(model_config).to(device)

# 파일에 적힌 글자 종류 수와 지금 토크나이저의 글자 종류 수가 같아야 번호가 맞는다. 데이터를 다시 만들고 학습을 안 했으면 여기서 멈춘다.
assert model_config.vocab_size == vocab_size, "모델 파일과 토크나이저의 글자 종류 수가 다르다 — train.py 로 다시 학습한다"

# 파일의 숫자를 빈 모델에 채워 넣는다 (문서 1-1 ②).
model.load_state_dict(ckpt["model"])
# 결과: model_config = ModelConfig(vocab_size=1244, block_size=128, n_embd=256, n_head=8, n_layer=6, dropout=0.2) — 파일에 적힌 그대로
#       <All keys matched successfully> — 숫자 5,410,012개가 빈 모델 칸에 그대로 들어감

# 글을 만들 때는 dropout 을 꺼야 한다 — 모델을 점검 모드로 바꾼다 (6단계 문서 4-3 ③).
# 8단계 조각 3 부터 dropout 이 실제로 켜져 있어서(0.2) 이 줄이 없으면 매번 숫자 20% 가 지워진 채 답한다 (8단계 문서 3-3 ③).
model.eval()

# 글을 이어 쓰게 할 시작 글. 학습 데이터와 같은 모양("사용자: ...\n봇:")이라 모델이 봇의 답을 이어 쓰게 된다.
prompt = "사용자: 배고파\n봇:"

# 시작 글을 모델이 받는 모양 (1, T) 의 번호 텐서로 만든다 (문서 1-2).
# encode(prompt) 는 길이 T 인 번호 리스트 하나를 돌려준다. 모델은 (B, T) — "문장 B개 묶음" 을 받는다.
# 문장이 한 개뿐이니 B = 1 이 되도록 리스트를 한 겹 더 감싼다. dtype 은 번호이니 long, device 는 모델과 같은 곳.
idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
# 결과: idx.shape = torch.Size([1, 11]) — 문장 1개 · 글자 11개 (시작 글 "사용자: 배고파" + 줄바꿈 + "봇:")

# 점수만 재는 일이라 기록(no_grad)을 끈다 (6단계 문서 4-3 ②). 모델에 넣으면 글자 자리마다 글자 종류 수만큼 점수가 나온다.
with torch.no_grad():
    logits = model(idx)
# 결과: logits.shape = torch.Size([1, 11, 1244]) — 글자 자리 11개마다 1244개 점수 (7단계 땐 69개)


# ---- 조각 2: 다음 글자 하나 뽑기 ----

# 점수 (1, 11, 1244) 에서 맨 마지막 글자 자리의 점수만 꺼낸다 (문서 2-1).
# 시작 글을 끝까지 읽은 뒤에 올 글자가 궁금하니 마지막 자리다. 묶음 칸은 그대로 · 글자 자리는 맨 끝 · 점수는 전부.
last = logits[:, -1, :]
# 결과: last.shape = torch.Size([1, 1244]) — 마지막 자리의 점수 1244개

# 점수를 확률로 바꾼다 — 0 이상 · 합이 1 (문서 2-2 ②). 점수가 늘어선 방향은 맨 끝 칸이다.
probs = F.softmax(last, dim=-1)
# 결과: probs.shape = torch.Size([1, 1244]) · probs.sum() = 1.0000004 (소수 계산 오차) · 공백 확률 0.99996

# 확률대로 글자 번호 하나를 뽑는다 (문서 2-2 ③). 확률 0.7 인 글자는 10번 중 약 7번 뽑힌다.
next_idx = torch.multinomial(probs, num_samples=1)
# 결과: next_idx.shape = torch.Size([1, 1]) · decode 하면 ' ' (공백) — "봇:" 뒤는 학습 데이터 10,637개가 전부 공백이라 거의 늘 같다


# ---- 조각 3: 한 글자씩 붙여 이어 쓰기 ----

# 조각 2 의 "점수 → 확률 → 뽑기" 를 max_new_tokens 번 되풀이하는 함수. 뽑을 때마다 입력 뒤에 붙인다 (문서 3-1).
# 위 조각 2 는 한 걸음을 풀어서 보여 준 것이고, 아래 함수 안의 같은 줄들이 실제로 되풀이된다.
@torch.no_grad()  # 점수만 재는 일이라 기록을 끈다 (조각 1 의 with torch.no_grad() 와 같은 일)
def generate(model, idx, max_new_tokens):
    """idx (1, T) 뒤에 글자를 max_new_tokens 개 이어 붙여 (1, T + max_new_tokens) 를 돌려준다."""
    for _ in range(max_new_tokens):
        # 모델은 block_size(128) 글자까지만 볼 수 있다 (문서 3-2). 모델에 줄 입력만 맨 끝 block_size 글자로 자른다.
        # 붙여 나가는 원본 idx 는 자르지 않는다. 글자가 모자라면 자를 것 없이 전부 남는다.
        # block_size 는 모델을 만든 model_config 에서 꺼낸다 — 모델의 위치 표 크기와 같은 값이라 어긋날 일이 없다.
        idx_cond = idx[:, -model_config.block_size:]

        logits = model(idx_cond)  # (1, T') -> (1, T', 1244)  (T' = 잘린 뒤 길이, 최대 128)
        last = logits[:, -1, :]  # (1, T', 1244) -> (1, 1244)  마지막 자리 점수 (조각 2)
        probs = F.softmax(last, dim=-1)  # (1, 1244) 점수 -> 확률
        next_idx = torch.multinomial(probs, num_samples=1)  # (1, 1) 뽑힌 글자 번호

        # 뽑은 글자를 idx 뒤에 이어 붙인다 (문서 3-1). (1, T) + (1, 1) -> (1, T + 1)
        # dim=1 은 글자 자리가 늘어선 방향(가로). dim=0 이면 묶음 칸 쪽으로 쌓여 문장이 2개가 된다.
        idx = torch.cat((idx, next_idx), dim=1)
    return idx
# 결과: generate(model, idx, 40).shape = torch.Size([1, 51]) · 200글자 이어 써도 torch.Size([1, 211]) (128 넘어도 에러 없음)
# 결과: 40글자 예 — 시작 글 뒤로 "봇: 울는 게 낫죠." 다음 "사용자: 우산 가고 싶어" · "봇: 부러워요." 처럼 대화가 이어짐 (뽑기라 실행마다 다름)
#       대화 모양(사용자/봇 줄 · 빈 줄)은 지키지만 답이 질문과 안 맞는 일이 많다 — 아래 reply 결과 참고

# ---- 조각 4: 대화해 보기 ----

def reply(model, text, max_new_tokens=60):
    """사용자가 한 말 text 에 모델이 봇으로서 답한 한 줄을 돌려준다. 예) "배고파" -> "아직 안 먹었어요"."""
    # 학습 데이터와 똑같은 모양의 시작 글을 만든다 (문서 4-1). 모양: "사용자: " + 말 + 줄바꿈 + "봇:"
    # 이 모양이어야 모델이 봇의 답을 이어 쓴다 (조각 1 의 prompt 와 같은 모양).
    prompt = f"사용자: {text}\n봇:"
    # 결과: text 가 "배고파" 면 prompt 는 조각 1 의 시작 글과 같다 (11글자)

    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)  # 조각 1: 시작 글 -> (1, T) 번호 텐서
    out = generate(model, idx, max_new_tokens)  # 조각 3: 시작 글 뒤에 max_new_tokens 글자를 이어 붙인다 -> (1, T + max_new_tokens)
    new_text = decode(out[0].tolist())[len(prompt):]  # 번호 -> 글, 시작 글 길이만큼 앞을 잘라 새로 만든 부분만 남긴다

    # 모델은 봇의 답이 끝나도 멈추지 않고 다음 "사용자:" 줄까지 계속 만든다 (조각 3 의 한계).
    # 학습 데이터에서 봇의 답은 한 줄이라 줄바꿈이 나오면 답이 끝난 것이다 (문서 4-2). 첫 줄바꿈 앞까지만 취한다.
    # split 은 글을 그 글자 기준으로 잘라 리스트로 돌려주고, [0] 은 맨 앞 조각이다.
    answer = new_text.split("\n")[0]
    # 결과: new_text 가 " 공부가 되길 바랍니다." 다음 줄바꿈 둘 · "사용자: ..." 로 이어져도 answer 는 " 공부가 되길 바랍니다."

    return answer.strip()  # "봇:" 바로 뒤가 공백이라 답이 공백으로 시작한다 — 앞뒤 공백을 걷어 낸다
    # 결과 (8단계 조각 3 · 6층 · 256칸 · 검증 1.64 모델 · torch.manual_seed(1)):
    #   시험 망쳤어 → 공부가 되길 바랍니다. · 날씨 좋아 → 좀 더 좋은 운동 하세요. · 배고파 → 로맨틱하네요. · 면허 따는 중 → 얼굴을 피해주세요.
    #   문장 모양은 자연스러워졌지만 질문과 안 맞는 답이 많다 · "hello" 는 encode 에서 KeyError('h') -> chat() 이 "모르는 글자가 있어요" 출력
    #   (7단계 토이 모델: 배고파 → 물풀을 먹었어요 — 질문 15종류를 외운 답)


def chat():
    """터미널에서 한 줄씩 말을 주고받는다. 그냥 엔터를 치면 끝난다. 이전 말은 기억하지 않는다 (매번 새 시작 글)."""
    print("봇과 대화합니다 (그냥 엔터를 치면 끝)")
    while True:
        text = input("사용자: ")
        if not text:
            break
        try:
            print("봇:", reply(model, text))
        except KeyError:  # encode 가 학습 글에 없는 글자(1244개 밖)를 만나면 KeyError — 영어 · 느낌표 등
            print("봇: 모르는 글자가 있어요")


# 직접 실행했을 때만 확인용 출력 (uv run python -m llm.generate)
if __name__ == "__main__":
    print(idx.shape)  # 기대: torch.Size([1, 11]) — 문장 1개 · 글자 11개 ("사용자: 배고파\n봇:")
    print(decode(idx[0].tolist()))  # 기대: 시작 글이 그대로 — 번호로 바꿨다 되돌려 제자리인지 확인
    print(logits.shape)  # 기대: torch.Size([1, 11, 1244]) — 글자 자리마다 1244개 점수
    print(last.shape)  # 기대: torch.Size([1, 1244]) — 마지막 자리의 점수 1244개
    print(probs.sum().item())  # 기대: 1.0 근처 — 확률은 합이 1
    print(next_idx.shape)  # 기대: torch.Size([1, 1]) — 뽑힌 글자 번호 1개
    print(repr(decode(next_idx[0].tolist())))  # 기대: ' ' (공백) — "봇:" 뒤는 공백이 거의 확실
    out = generate(model, idx, max_new_tokens=40)
    print(out.shape)  # 기대: torch.Size([1, 51]) — 시작 글 11글자 + 새 글자 40개
    print(decode(out[0].tolist()))  # 기대: 시작 글 뒤에 40글자가 이어진 글 (뽑기라 실행마다 다를 수 있다)
    print(generate(model, idx, max_new_tokens=200).shape)  # 기대: torch.Size([1, 211]) — 128 글자를 넘겨도 에러 없이 (자르기가 돌아야 통과)
    a = reply(model, "배고파")
    print(repr(a))  # 기대: 줄바꿈 없는 짧은 한 줄, 예) '아직 안 먹었어요' (뽑기라 실행마다 다를 수 있다)
    print("\n" in a)  # 기대: False — 답이 한 줄로 끊겼는지
    chat()  # 직접 말해 보기 — 그냥 엔터를 치면 끝
