"""학습 데이터 준비.

사람이 쓴 공개 한국어 대화(songys/Chatbot_data · 질문 1 · 답 1 짝 11,823개 · MIT 라이선스)를 내려받아
2단계와 같은 모양("사용자: 질문 / 봇: 답")의 data/train.txt (학습용) · data/val.txt (검증용) 로 바꿔 저장한다.
2단계의 템플릿 생성기는 git 기록에 남아 있다. 데이터를 바꾼 이유는 결정 기록 0012.

8단계 조각 1: 원본 내려받기 (문서 1-1) → CSV 읽기 (문서 1-2) → 섞기 (문서 1-3).
8단계 조각 2: 같은 질문끼리 묶어 질문 단위로 섞고 나누기 → 두 파일로 쓰기 (문서 2-1 · 2-2).
"""

# csv: 쉼표로 칸을 나눈 표 모양 파일(CSV)을 읽는 표준 라이브러리. 칸 안의 쉼표 · 따옴표를 알아서 처리한다 (문서 1-2).
import csv

# random: 질문 순서를 섞는 데 쓴다 (문서 1-3 · 2-2).
import random

# urllib.request: 인터넷 주소(URL)에서 파일을 받아 오는 표준 라이브러리. 따로 설치할 것이 없다 (문서 1-1).
import urllib.request

# pathlib.Path: 파일 경로를 객체로 다룬다. 윈도우(\) · 맥/리눅스(/) 구분자 차이를 알아서 처리한다 (2단계 문서 8번).
from pathlib import Path

# 원본 CSV 의 인터넷 주소. GitHub 리포의 파일을 "내용 그대로" 보여 주는 raw 주소다.
URL = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"

# 이 파일(make_dataset.py)이 든 폴더 = data/. 어느 폴더에서 실행하든 같은 자리를 가리킨다.
DATA_DIR = Path(__file__).parent

# 받은 원본을 그대로 두는 곳. 가공한 글과 섞이지 않게 하위 폴더 raw/ 에 둔다.
# data/raw/ · data/train.txt · data/val.txt 는 .gitignore 대상 — 남의 데이터 복사본은 리포에 올리지 않고 이 코드만 올린다 (문서 1-1 ④).
RAW_PATH = DATA_DIR / "raw" / "ChatbotData.csv"
# 학습용 · 검증용 글을 파일 둘로 나눠 쓴다 (문서 2-2 ④).
TRAIN_PATH = DATA_DIR / "train.txt"
VAL_PATH = DATA_DIR / "val.txt"


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


def split_by_question(pairs: list[tuple[str, str]]) -> tuple[list, list]:
    """대화 짝들을 질문 단위로 묶어 섞은 뒤 (학습용 짝들, 검증용 짝들) 로 나눈다 (문서 2-1 · 2-2).

    같은 질문의 대화는 반드시 같은 쪽으로 간다 — 검증용 질문이 학습용에 한 번도 나오지 않게(데이터 누수 막기).
    """
    # 같은 질문의 대화를 한 묶음으로 모은다. 사전 하나 — 이름표(키) = 질문 · 값 = 그 질문의 대화 목록.
    # 원본에는 같은 질문이 두 번 이상 나오는 경우가 156개 있다 (예: "고양이 키우고 싶어" 에 답이 둘).
    groups = {}
    for q, a in pairs:
        # setdefault(키, 기본값): 키가 있으면 그 값을, 없으면 기본값(빈 목록)을 넣고 그 값을 돌려준다.
        # 돌려받은 목록에 바로 .append 로 대화를 넣는다 — "없으면 빈 목록부터 만들고 넣기" 를 한 줄로.
        # 이름표는 질문 q — 같은 질문끼리 같은 묶음이어야 한쪽(학습용 또는 검증용)으로 같이 간다 (문서 2-2 ③).
        groups.setdefault(q, []).append((q, a))
    # 결과: len(groups) = 11662 (대화 11,823개 → 질문 묶음 11,662개)
    #       groups["고양이 키우고 싶어"] = [('고양이 키우고 싶어', '자신을 먼저 키우세요.'), ('고양이 키우고 싶어', '가족들과 상의해보세요.')]

    # 묶음의 이름표(질문)만 꺼내 목록으로. 사전을 list() 로 감싸면 키 목록이 된다.
    questions = list(groups)

    # 원본이 주제순으로 정렬돼 있어서 섞는다 (문서 1-3 — 이유는 그대로, 섞는 단위만 대화 → 질문).
    # seed 0 고정: 실행마다 같은 순서 · 같은 파일 (2단계와 같은 이유).
    random.seed(0)
    # 섞는 것은 대화 짝(pairs)이 아니라 질문 목록 — 질문 묶음 단위로 나눌 것이라서 (문서 2-2 ④).
    random.shuffle(questions)

    # 질문의 10% 를 검증용으로. // 는 나눈 몫만 남기는 나눗셈 (질문 수는 정수여야 한다).
    n_val = len(questions) // 10
    # 앞 n_val 개는 검증용 · 나머지는 학습용. 슬라이스 [시작:끝] — 비우면 처음부터 · 끝까지 (5단계 문서 1-3).
    # 학습용은 n_val 번째부터 끝까지 — 검증용 [:n_val] 과 겹치지도 빠지지도 않는다.
    val_questions = questions[:n_val]
    train_questions = questions[n_val:]
    # 결과: n_val = 1166 · len(train_questions) = 10496 (합 11,662)

    # 질문 목록을 다시 대화 짝 목록으로 편다. 질문마다 그 묶음의 대화를 차례로 꺼낸다.
    # [식 for 바깥 in ... for 안쪽 in ...] — for 를 두 번 쓴 컴프리헨션. 바깥(질문)마다 안쪽(그 질문의 대화)을 모두 꺼내 한 줄 리스트로.
    train_pairs = [pair for q in train_questions for pair in groups[q]]
    val_pairs = [pair for q in val_questions for pair in groups[q]]
    # 결과: len(train_pairs) = 10637 · len(val_pairs) = 1186 (합 11,823 — 대화 하나도 안 빠짐)
    #       val_pairs[0] = ('면허 따는 중', '안전 운전 하세요.')
    return train_pairs, val_pairs


def write(path: Path, pairs: list[tuple[str, str]]) -> str:
    """대화 짝들을 2단계와 같은 모양의 글로 만들어 path 에 쓰고, 그 글을 돌려준다."""
    # 짝마다 두 줄 대화로 만들고, 사이에 빈 줄 하나를 두고 이어 붙인다 (2단계 문서 6번 ⑤).
    # 모양이 2단계와 같아야 tokenizer · dataset · train · generate 를 크게 고치지 않고 쓸 수 있다.
    text = "\n".join(f"사용자: {q}\n봇: {a}\n" for q, a in pairs)
    # utf-8 로 저장한다 (윈도우 기본 cp949 로 저장되면 한글이 깨진다).
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main() -> None:
    download()
    pairs = load_pairs()

    # 질문 단위로 나눠 학습용 · 검증용 파일을 따로 쓴다 (문서 2-2 ④).
    # 파일이 둘이라 dataset.py 가 글자 수 90% 지점에서 자를 필요가 없어진다 — 경계에서 대화가 잘리지 않는다.
    train_pairs, val_pairs = split_by_question(pairs)
    train_text = write(TRAIN_PATH, train_pairs)
    val_text = write(VAL_PATH, val_pairs)

    # 결과 요약. 글자 종류는 두 글을 합쳐 센다 — 토크나이저도 그렇게 센다 (문서 2-3).
    # 검증용 질문 중 학습용에도 있는 것의 개수 = 누수. 0 이어야 한다 (문서 2-1).
    # {식 for ...} 는 집합(중복 없는 모음)을 한 줄로 만드는 문법 · 집합 & 집합 = 양쪽에 다 있는 것(교집합).
    leaked = {q for q, _ in val_pairs} & {q for q, _ in train_pairs}
    print(f"학습용 {len(train_pairs)}개 · {len(train_text)}자 → {TRAIN_PATH}")
    print(f"검증용 {len(val_pairs)}개 · {len(val_text)}자 → {VAL_PATH}")
    print(f"글자 종류 {len(set(train_text + val_text))}개 · 검증용에만 있는 글자 {len(set(val_text) - set(train_text))}개 · 겹친 질문 {len(leaked)}개")
    # 결과 (터미널에 출력):
    #   학습용 10637개 · 413139자 → C:\...\data\train.txt
    #   검증용 1186개 · 46705자 → C:\...\data\val.txt
    #   글자 종류 1244개 · 검증용에만 있는 글자 22개 · 겹친 질문 0개
    #   (조각 1 방식 — 글 하나를 90% 에서 자르기 — 일 때는 겹친 질문 32개 · 경계에서 잘린 대화 1개)
    # val.txt 앞부분:
    #   사용자: 면허 따는 중
    #   봇: 안전 운전 하세요.


# 직접 실행했을 때만 main() 을 부른다. 실행: uv run python data/make_dataset.py
if __name__ == "__main__":
    main()
