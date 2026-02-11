# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the web application
streamlit run src/ui/app.py

# Run CLI fallback
python main.py

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/unit/test_spaced_repetition.py -v

# Run a single test function
pytest tests/unit/test_spaced_repetition.py::test_sm2_algorithm -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run unit tests only
pytest tests/unit/ -v

# Run E2E tests (requires app running on :8501)
python -m streamlit run src/ui/app.py --server.headless true &
pytest tests/e2e/ -v --base-url http://localhost:8501 --browser chromium

# Full validation pipeline (starts app automatically if not running)
./scripts/validate.sh

# Unit tests only via script
./scripts/validate.sh --unit

# E2E tests only via script (auto-starts app)
./scripts/validate.sh --e2e

# Run linting
pylint src/

# Type checking
mypy src/
```

## Architecture

This is a Streamlit-based AI Chinese learning application targeting HSK 1-2 learners.

### Module Structure

```
src/
├── ai/
│   ├── ai_tutor.py          # Claude API integration for conversation, grammar explanation
│   └── agents.py            # Multi-agent system: Orchestrator, Content, Tutor, Eval, Data
├── core/
│   ├── data_parser.py       # Chinese vocabulary parsing, pinyin conversion, HSK word loading
│   ├── lesson_manager.py    # Lesson creation (10 words/lesson), quiz generation
│   └── progress_tracker.py  # SQLite DB manager (6 tables), statistics, SRS review queue
├── learning/
│   ├── spaced_repetition.py # SM-2 algorithm implementation for review scheduling
│   └── gamification.py      # XP system, level calculation (1.5x scaling), achievements
├── speech/
│   └── speech_handler.py    # Google TTS audio generation with file cache
└── ui/
    └── app.py               # Streamlit multi-page app (Home, Vocab, Chat, Quiz, Progress)
```

### Data Flow

1. `data/vocabulary.json` → `ChineseDataParser.load_hsk_words()` → vocabulary list
2. `LessonManager` slices vocabulary into 10-word lessons and generates quizzes
3. `ProgressTracker` (SQLite) stores sessions, word_mastery, pronunciation_history, conversation_practice, user_progress, achievements
4. `SpacedRepetition` uses SM-2 algorithm to schedule word reviews from word_mastery table
5. `Gamification` reads from user_progress table to award XP, update levels, unlock achievements
6. `ChineseAITutor` wraps Claude API; `agents.py` adds Orchestrator pattern on top

### Database (SQLite)

Six tables in `database/learning_progress.db`:
- `sessions` — study session records
- `word_mastery` — per-word SM-2 fields (easiness_factor, interval, repetitions, next_review)
- `pronunciation_history` — TTS practice logs
- `conversation_practice` — AI chat turns with corrections
- `user_progress` — singleton row: level, XP, streak, daily goal
- `achievements` — unlockable badges by category (words, streak, score, time, special)

### Multi-Agent System (`src/ai/agents.py`)

```
OrchestratorAgent
├── ContentAgent     # generates lessons and dialogues via Claude
├── TutorAgent       # chat responses with grammar corrections
├── EvalAgent        # quiz generation, adaptive difficulty
└── DataAgent        # progress analysis, retention predictions
```

The Orchestrator classifies user intent and delegates to the appropriate agent.

### Key Algorithms

- **SM-2** (`spaced_repetition.py`): quality 0-5 → updates easiness_factor, interval, next_review date
- **Level XP** (`gamification.py`): XP(n) = 100 × 1.5^(n-1); adaptive difficulty based on recent 10 scores
- **Quiz types**: translation (Chinese→Korean), listening (play audio→choose), fill-in-blank

### Environment Variables

```
ANTHROPIC_API_KEY   # Required for AI tutor and agent features
DATABASE_PATH       # Default: database/learning_progress.db
AUDIO_CACHE_DIR     # Default: audio_cache
```

All features degrade gracefully when API key is absent (fallback responses used).
