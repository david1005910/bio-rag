# 중국어 학습 프로그램 - 설치 및 실행 가이드

## 📋 사전 요구사항

- Python 3.8 이상
- pip (Python 패키지 관리자)
- 최소 500MB의 여유 디스크 공간

## 🚀 설치 방법

### 1. 프로젝트 다운로드

이메일로 받은 압축 파일을 압축 해제하거나, GitHub에서 클론:

```bash
# GitHub에서 클론 (향후)
git clone https://github.com/yourusername/chinese-learning-app.git
cd chinese-learning-app
```

### 2. 가상환경 생성 (권장)

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

**주의**: 일부 패키지 설치가 실패할 수 있습니다. 다음은 선택적 패키지입니다:
- `pyaudio`: 마이크 입력 (설치 어려울 경우 생략 가능)
- `anthropic`: AI 튜터 (API 키 필요)

### 4. 환경변수 설정 (선택사항)

AI 튜터 기능을 사용하려면:

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일을 편집하여 API 키 입력
# ANTHROPIC_API_KEY=your_actual_api_key_here
```

Anthropic API 키는 https://console.anthropic.com/ 에서 발급받을 수 있습니다.

### 5. 디렉토리 구조 확인

다음 디렉토리들이 자동으로 생성됩니다:
- `database/`: 학습 진도 저장
- `audio_cache/`: 음성 파일 캐시

## 🎮 실행 방법

### 웹 인터페이스 (추천)

```bash
streamlit run src/ui/app.py
```

브라우저에서 자동으로 열립니다. 열리지 않으면 http://localhost:8501 접속

### CLI 인터페이스

```bash
python main.py
```

## 🔧 문제 해결

### 1. "No module named 'xxx'" 오류

```bash
# 해당 패키지 개별 설치
pip install 패키지이름
```

### 2. pyaudio 설치 실패 (Windows)

```bash
# 미리 컴파일된 wheel 파일 사용
pip install pipwin
pipwin install pyaudio
```

또는 발음 평가 기능 없이 사용 가능합니다.

### 3. streamlit 실행 오류

```bash
# streamlit 재설치
pip uninstall streamlit
pip install streamlit
```

### 4. API 키 오류

- `.env` 파일이 올바른 위치에 있는지 확인
- API 키가 정확한지 확인
- AI 기능은 선택사항이므로 없어도 다른 기능 사용 가능

### 5. 데이터베이스 오류

```bash
# 데이터베이스 재생성
rm database/learning_progress.db
# 앱 재실행하면 자동 생성됨
```

## 📚 사용 방법

### 기본 워크플로우

1. **웹 앱 실행**
   ```bash
   streamlit run src/ui/app.py
   ```

2. **단어 학습**
   - 왼쪽 메뉴에서 "단어 학습" 선택
   - 레슨 번호 입력
   - 🔊 버튼으로 발음 듣기
   - 외운 단어 체크

3. **회화 연습**
   - "회화 연습" 메뉴 선택
   - 중국어로 입력
   - AI가 응답 및 교정

4. **퀴즈**
   - "퀴즈" 메뉴 선택
   - 문제 풀기
   - 즉시 채점

5. **진도 확인**
   - "진도 확인" 메뉴에서 통계 보기

## 🎯 학습 팁

1. **매일 조금씩**: 하루 10-15분씩 꾸준히
2. **복습 중요**: 간격 반복 시스템 활용
3. **발음 연습**: 음성을 여러 번 들어보기
4. **AI 대화**: 실수를 두려워하지 말고 대화하기
5. **목표 설정**: 주간 목표 설정 (예: 50개 단어 마스터)

## 📊 데이터 백업

학습 데이터는 로컬에 저장됩니다:

```bash
# 데이터베이스 백업
cp database/learning_progress.db backup_$(date +%Y%m%d).db
```

## 🆘 지원

문제가 있으면 다음을 확인하세요:

1. 문서: `docs/` 폴더의 PRD.md, TRD.md
2. 로그 파일: 오류 메시지 확인
3. GitHub Issues (향후 공개 시)

## 🔄 업데이트

```bash
# 최신 코드 받기
git pull

# 의존성 업데이트
pip install -r requirements.txt --upgrade
```

## 📝 추가 정보

- **HSK 레벨**: 현재 HSK 1급 단어 지원
- **언어**: 중국어 간체자 기준
- **오프라인**: 기본 기능은 오프라인 사용 가능 (AI 제외)

## ⚠️ 주의사항

- API 키는 절대 공개하지 마세요
- 개인 학습 데이터는 로컬에만 저장됩니다
- 음성 파일 캐시는 주기적으로 정리하세요

즐거운 중국어 학습 되세요! 🎉
