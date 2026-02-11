# 🇨🇳 중국어 학습 프로그램

AI 기반 개인화된 중국어 학습 플랫폼

## ✨ 주요 기능

- 📚 **단어 학습**: HSK 1-2급 기반 체계적인 어휘 학습
- 💬 **AI 회화**: Claude API를 활용한 실시간 대화 연습
- 🎤 **발음 평가**: 음성 인식 기반 발음 정확도 측정
- 🎵 **성조 분석**: 중국어 4성조 시각화 및 교정
- 📝 **퀴즈**: 학습 내용 확인 및 복습
- 🔄 **간격 반복**: SM-2 알고리즘 기반 효율적 복습
- 🎮 **게임화**: 레벨, 업적, 연속 학습 스트릭
- 📊 **진도 관리**: 학습 통계 및 시각화

## 🚀 빠른 시작

### 사전 요구사항

- Python 3.8 이상
- pip
- Anthropic API 키 (선택사항)

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/yourusername/chinese-learning-app.git
cd chinese-learning-app

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 ANTHROPIC_API_KEY를 입력하세요
```

### 실행

#### 웹 인터페이스 (Streamlit)

```bash
streamlit run src/ui/app.py
```

브라우저에서 http://localhost:8501 접속

#### CLI 인터페이스

```bash
python main.py
```

## 📖 사용법

### 1. 단어 학습

- 레슨을 선택하고 단어를 학습합니다
- 🔊 버튼을 눌러 발음을 들을 수 있습니다
- 외운 단어는 체크박스로 표시합니다

### 2. 회화 연습

- AI 튜터와 중국어로 대화를 나눕니다
- 문법 오류가 있으면 자동으로 교정됩니다
- 더 자연스러운 표현을 제안받을 수 있습니다

### 3. 발음 연습

- 목표 문장을 보고 녹음합니다
- AI가 발음을 평가하고 점수를 줍니다
- 개선 피드백을 받을 수 있습니다

### 4. 퀴즈

- 학습한 내용으로 퀴즈를 풉니다
- 즉시 채점 및 해설을 확인합니다

### 5. 진도 확인

- 학습 시간, 마스터한 단어 수 등 통계를 확인합니다
- 학습 곡선 그래프로 진도를 시각화합니다

## 📁 프로젝트 구조

```
chinese-learning-app/
├── src/
│   ├── core/           # 핵심 로직
│   ├── ai/             # AI 튜터
│   ├── speech/         # 음성 처리
│   ├── learning/       # 학습 시스템 (SRS, 게임화)
│   └── ui/             # 웹 UI
├── data/               # 학습 데이터
├── database/           # SQLite 데이터베이스
├── tests/              # 테스트
├── docs/               # 문서
└── requirements.txt    # 의존성
```

## 🛠️ 기술 스택

- **언어**: Python 3.8+
- **AI**: Anthropic Claude API
- **음성**: Google TTS/STT, librosa
- **데이터베이스**: SQLite
- **웹 프레임워크**: Streamlit
- **시각화**: Plotly, Matplotlib

## 📊 데이터 소스

- **단어**: CC-CEDICT (Creative Commons)
- **HSK 단어**: 공식 HSK 단어 목록
- **예문**: Tatoeba 프로젝트

## 🧪 테스트

```bash
# 모든 테스트 실행
pytest

# 커버리지와 함께 실행
pytest --cov=src tests/
```

## 📝 문서

- [PRD (Product Requirements Document)](docs/PRD.md)
- [TRD (Technical Requirements Document)](docs/TRD.md)

## 🤝 기여

기여를 환영합니다! Pull Request를 보내주세요.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 라이선스

MIT License

## 📧 연락처

프로젝트 링크: https://github.com/yourusername/chinese-learning-app

## 🙏 감사의 말

- CC-CEDICT 프로젝트
- Anthropic Claude
- Tatoeba 프로젝트
- 모든 오픈소스 기여자들
