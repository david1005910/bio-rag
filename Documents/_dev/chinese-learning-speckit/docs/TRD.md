# Technical Requirements Document (TRD)
# AI 기반 중국어 학습 프로그램

## 1. 시스템 아키텍처

### 1.1 모듈 구조

```
chinese-learning-app/
├── data/                       # 데이터 파일
│   ├── vocabulary.json         # 단어 데이터
│   ├── grammar_rules.json      # 문법 규칙
│   └── dialogues.json          # 회화 예문
│
├── src/                        # 소스 코드
│   ├── core/                   # 핵심 로직
│   │   ├── data_parser.py     # 데이터 파싱
│   │   ├── lesson_manager.py  # 레슨 관리
│   │   ├── quiz_generator.py  # 퀴즈 생성
│   │   └── progress_tracker.py # 진도 추적
│   │
│   ├── ai/                     # AI 관련
│   │   └── ai_tutor.py        # AI 튜터
│   │
│   ├── speech/                 # 음성 처리
│   │   ├── speech_handler.py  # TTS
│   │   ├── speech_recognition_handler.py # STT
│   │   └── tone_analyzer.py   # 성조 분석
│   │
│   ├── learning/               # 학습 시스템
│   │   ├── spaced_repetition.py # SRS
│   │   └── gamification.py    # 게임화
│   │
│   └── ui/                     # 사용자 인터페이스
│       ├── app.py             # Streamlit 메인 앱
│       └── enhanced_app.py    # 고급 기능 앱
│
├── database/                   # 데이터베이스
│   └── learning_progress.db   # SQLite DB
│
├── audio_cache/                # 음성 파일 캐시
│
├── tests/                      # 테스트
│
├── docs/                       # 문서
│   ├── PRD.md                 # 제품 요구사항
│   └── TRD.md                 # 기술 요구사항
│
├── requirements.txt            # Python 의존성
├── README.md                   # 프로젝트 소개
└── main.py                     # CLI 진입점
```

## 2. 기술 스택

### 2.1 핵심 라이브러리

```txt
# Core
python>=3.8

# Data Processing
pandas==2.0.3
numpy==1.24.3
pypinyin==0.49.0

# AI & NLP
anthropic==0.18.1

# Speech Processing
gtts==2.5.0
playsound==1.3.0
SpeechRecognition==3.10.0
librosa==0.10.1
soundfile==0.12.1

# Web Interface
streamlit==1.29.0

# Visualization
plotly==5.18.0
matplotlib==3.8.2

# Utils
python-dotenv==1.0.0

# Testing
pytest==7.4.3
```

## 3. 데이터베이스 설계

### 3.1 ERD

```
sessions (학습 세션)
├── id (PK)
├── start_time
├── end_time
├── lesson_number
├── words_learned
└── quiz_score

word_mastery (단어 마스터)
├── word_id (PK)
├── simplified
├── traditional
├── pinyin
├── times_practiced
├── times_correct
├── last_practiced
├── mastery_level
├── next_review
├── easiness_factor
├── interval
└── repetitions

pronunciation_history (발음 기록)
├── id (PK)
├── word
├── score
├── attempt_time
└── recognized_text

user_progress (사용자 진도)
├── id (PK)
├── level
├── xp
└── total_xp

achievements (업적)
├── achievement_id (PK)
├── unlocked
└── unlocked_at

streak_data (연속 학습)
├── id (PK)
├── current_streak
├── longest_streak
└── last_study_date
```

## 4. API 통합

### 4.1 Claude API

```python
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2000,
    messages=[
        {"role": "user", "content": prompt}
    ]
)
```

## 5. 간격 반복 학습 (SRS)

### 5.1 SM-2 알고리즘

```python
def calculate_next_interval(card, quality):
    """
    Quality Scale:
        0: Complete blackout
        1: Incorrect, but familiar
        2: Incorrect, but easy to recall
        3: Correct, but difficult
        4: Correct, with hesitation
        5: Perfect response
    """
    if quality < 3:
        card.repetitions = 0
        card.interval = 1
    else:
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = round(
                card.interval * card.easiness_factor
            )
        card.repetitions += 1
    
    # Update easiness factor
    card.easiness_factor = max(1.3, 
        card.easiness_factor + 
        (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    )
    
    return card
```

## 6. 성능 최적화

### 6.1 데이터베이스 인덱스

```sql
CREATE INDEX idx_word_mastery_next_review 
ON word_mastery(next_review);

CREATE INDEX idx_sessions_start_time 
ON sessions(start_time);
```

### 6.2 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_word_definition(word: str) -> dict:
    """단어 정의 캐싱"""
    pass
```

## 7. 보안

### 7.1 API 키 관리

```python
# .env 파일
ANTHROPIC_API_KEY=sk-ant-xxxxx

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get("ANTHROPIC_API_KEY")
```

## 8. 테스트

### 8.1 단위 테스트

```python
def test_calculate_next_interval_quality_0():
    card = Card(repetitions=5, interval=30)
    srs = SpacedRepetitionSystem(None)
    updated = srs.calculate_next_interval(card, quality=0)
    
    assert updated.repetitions == 0
    assert updated.interval == 1
```

## 9. 배포

### 9.1 로컬 설치

```bash
# 1. 저장소 클론
git clone https://github.com/username/chinese-learning-app.git
cd chinese-learning-app

# 2. 가상환경 생성
python -m venv venv
source venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경변수 설정
cp .env.example .env

# 5. 실행
streamlit run src/ui/app.py
```
