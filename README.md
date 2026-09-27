# LLM_Study_Project

작은 언어모델을 밑바닥부터 직접 구현하며 공부하는 프로젝트. [guppylm](https://github.com/arman-bd/guppylm) 참고.

포트폴리오용 완성품이 아니라 Mario_RL_Project 처럼 공부 기록.

## 왜

트랜스포머 내부를 블랙박스로 안 두고, 토크나이저부터 학습 루프까지 전 과정을 직접 짜보면서
각 구성요소가 왜 필요한지 · 다른 선택지와 뭐가 다른지 이해하는 것이 목적.

## 순서

1. `config.py` — 모델·학습 하이퍼파라미터
2. 토이 데이터셋 — 아주 작은 학습용 텍스트
3. `tokenizer.py` — 텍스트 ↔ 토큰 변환
4. `model.py` — 트랜스포머 아키텍처 (임베딩 → attention → FFN → 조립)
5. `dataset.py` — 데이터 로딩·배치
6. `train.py` — 학습 루프
7. `generate.py` — 추론·생성

각 단계는 코드를 같이 쓰면서 진행 — 한 번에 완성해서 던지지 않음.

## 환경

로컬 Python (GPU 없으면 CPU로 아주 작은 모델/데이터로 진행).
