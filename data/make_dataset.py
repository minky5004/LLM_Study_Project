"""학습 데이터 준비.

사람이 쓴 공개 한국어 대화(songys/Chatbot_data · 질문 1 · 답 1 짝 11,823개 · MIT 라이선스)를 내려받아
2단계와 같은 모양("사용자: 질문 / 봇: 답")의 data/train.txt 로 바꿔 저장한다.
2단계의 템플릿 생성기는 git 기록에 남아 있다. 데이터를 바꾼 이유는 결정 기록 0012.

지금은 8단계 조각 1까지: 원본 내려받기 (문서 1-1) → CSV 읽기 (문서 1-2) → 섞기 (문서 1-3) → train.txt 쓰기.
"""

# csv: 쉼표로 칸을 나눈 표 모양 파일(CSV)을 읽는 표준 라이브러리. 칸 안의 쉼표 · 따옴표를 알아서 처리한다 (문서 1-2).
import csv

# random: 대화 순서를 섞는 데 쓴다 (문서 1-3).
import random

# urllib.request: 인터넷 주소(URL)에서 파일을 받아 오는 표준 라이브러리. 따로 설치할 것이 없다 (문서 1-1).
import urllib.request

# pathlib.Path: 파일 경로를 객체로 다룬다. 윈도우(\) · 맥/리눅스(/) 구분자 차이를 알아서 처리한다 (2단계 문서 8번).
from pathlib import Path

# 원본 CSV 의 인터넷 주소. GitHub 리포의 파일을 "내용 그대로" 보여 주는 raw 주소다.
URL = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"

# 이 파일(make_dataset.py)이 든 폴더 = data/. 어느 폴더에서 실행하든 같은 자리를 가리킨다.
DATA_DIR = Path(__file__).parent

# 받은 원본을 그대로 두는 곳. 가공한 train.txt 와 섞이지 않게 하위 폴더 raw/ 에 둔다.
# data/raw/ 와 data/train.txt 는 .gitignore 대상 — 남의 데이터 복사본은 리포에 올리지 않고 이 코드만 올린다 (문서 1-1 ④).
RAW_PATH = DATA_DIR / "raw" / "ChatbotData.csv"
OUT_PATH = DATA_DIR / "train.txt"


def download() -> None:
    """원본 CSV 를 처음 한 번만 받아 RAW_PATH 에 저장한다 (문서 1-1).

    이미 받아 둔 파일이 있으면 아무것도 하지 않는다 — 매번 받으면 느리고, 인터넷이 끊기면 실행이 멈추기 때문이다.
    """
    # 파일이 "없을 때만" 받는다. exists() 는 그 경로에 파일이 있으면 True (문서 1-1 ③).
    if not RAW_PATH.exists():
        # raw/ 폴더가 없으면 만든다. parents=True: 중간 폴더까지 · exist_ok=True: 이미 있어도 에러 없이 넘어감.
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

        # URL 의 파일을 받아 RAW_PATH 에 저장한다. 괄호 안 순서는 (어디서 받을지, 어디에 저장할지).
        urllib.request.urlretrieve(URL, RAW_PATH)
        # 결과: data/raw/ChatbotData.csv 889,842 바이트 · 두 번째 실행부터는 이 블록을 건너뛴다


def load_pairs() -> list[tuple[str, str]]:
    """원본 CSV 를 읽어 (질문, 답) 짝의 리스트로 돌려준다 (문서 1-2).

    반환 예: [("12시 땡!", "하루가 또 가네요."), ("1지망 학교 떨어졌어", "위로해 드립니다."), ...]  — 11,823개
    """
    # encoding="utf-8": 한글이 깨지지 않게 (2단계 문서 7번).
    # newline="": csv 모듈 사용법이 요구하는 설정. 줄바꿈 처리를 csv 모듈에 맡긴다 (안 주면 칸 안 줄바꿈이 섞일 때 잘못 읽힐 수 있다).
    with open(RAW_PATH, encoding="utf-8", newline="") as f:
        # DictReader: 첫 줄(Q,A,label)을 칸 이름으로 삼고, 나머지 한 줄을 {"Q": ..., "A": ..., "label": ...} 사전 하나로 준다.
        reader = csv.DictReader(f)

        # 한 줄(row)에서 질문 칸 "Q" · 답 칸 "A" 의 값을 꺼내 짝으로 만든다. 칸 이름은 CSV 첫 줄(Q,A,label)에 있다 (문서 1-2 ①).
        # label(주제 번호)은 이번에 쓰지 않는다.
        # [식 for 변수 in 반복할것] — 반복하며 결과를 리스트에 담는 한 줄 문법(리스트 컴프리헨션, 2단계 문서 6번).
        pairs = [(row["Q"], row["A"]) for row in reader]
        # 결과: len(pairs) = 11823 · pairs[0] = ('12시 땡!', '하루가 또 가네요.')
        #       답 안 쉼표도 한 칸으로 읽힘 — '저를 만들어 준 사람을 부모님, 저랑 이야기해 주는 사람을 친구로 생각하고 있어요'

    return pairs


def main() -> None:
    download()
    pairs = load_pairs()

    # 원본은 주제 순서(일상 → 이별 → 사랑)로 정렬돼 있다. 그대로 이어 붙이면 dataset.py 가 자르는 뒤 10%(검증용)가 전부 사랑 이야기가 된다.
    # 대화 짝 단위로 섞어 세 주제가 앞뒤에 고루 들어가게 한다 (문서 1-3).
    # seed 0 고정: 실행마다 같은 순서로 섞여 같은 train.txt 가 나온다 (2단계와 같은 이유).
    random.seed(0)
    # 리스트를 제자리에서 무작위로 섞는다. 아무것도 돌려주지 않으니 결과를 받지 않는다 (문서 1-3 ③).
    random.shuffle(pairs)
    # 결과: pairs[0] = ('주말에 늦잠자고 브런치 먹으면서 수다 떨고 싶어', '완벽한 주말이네요.')
    #       dataset.py 가 자르는 뒤 10% 의 주제 — 일상 526 · 이별 362 · 사랑 295 (안 섞으면 전부 사랑)

    # 짝마다 2단계와 같은 두 줄 대화로 만들고, 사이에 빈 줄 하나를 두고 이어 붙인다 (2단계 문서 6번 ⑤).
    # 모양이 2단계와 같아야 tokenizer · dataset · train · generate 를 고치지 않고 그대로 쓸 수 있다.
    text = "\n".join(f"사용자: {q}\n봇: {a}\n" for q, a in pairs)

    # utf-8 로 저장한다 (윈도우 기본 cp949 로 저장되면 한글이 깨진다).
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    # 결과 요약: 대화 개수 · 전체 글자 수 · 글자 종류. 글자 종류 수가 곧 글자 단위 토크나이저의 vocab_size 다.
    print(f"저장: {OUT_PATH}")
    print(f"대화 {len(pairs)}개 · 글자 {len(text)}자 · 글자 종류 {len(set(text))}개")
    # 결과 (터미널에 출력):
    #   저장: C:\...\LLM_Study_Project\data\train.txt
    #   대화 11823개 · 글자 459845자 · 글자 종류 1244개   (2단계 토이 데이터: 3000개 · 65572자 · 69개)
    # train.txt 앞부분:
    #   사용자: 주말에 늦잠자고 브런치 먹으면서 수다 떨고 싶어
    #   봇: 완벽한 주말이네요.
    #   (빈 줄)
    #   사용자: 농구해야지
    #   봇: 운동은 뭐든 좋아요.


# 직접 실행했을 때만 main() 을 부른다. 실행: uv run python data/make_dataset.py
if __name__ == "__main__":
    main()
