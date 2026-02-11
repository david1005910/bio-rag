"""
학습 진도 추적 모듈
데이터베이스 관리 및 통계 (6 tables: sessions, word_mastery, pronunciation_history,
conversation_practice, user_progress, achievements)
"""

import sqlite3
import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple


class ProgressTracker:
    """학습 진도 추적 및 데이터베이스 관리"""

    def __init__(self, db_path: str = 'database/learning_progress.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = None
        self.setup_database()

    def setup_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()

        cursor.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                lesson_number INTEGER,
                words_learned INTEGER DEFAULT 0,
                quiz_score REAL,
                session_type TEXT DEFAULT 'lesson'
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);

            CREATE TABLE IF NOT EXISTS word_mastery (
                word_id INTEGER PRIMARY KEY AUTOINCREMENT,
                simplified TEXT UNIQUE,
                traditional TEXT,
                pinyin TEXT,
                definitions TEXT,
                times_practiced INTEGER DEFAULT 0,
                times_correct INTEGER DEFAULT 0,
                last_practiced TIMESTAMP,
                first_learned TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mastery_level TEXT DEFAULT 'new',
                next_review TIMESTAMP,
                easiness_factor REAL DEFAULT 2.5,
                interval INTEGER DEFAULT 0,
                repetitions INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_word_mastery_next_review ON word_mastery(next_review);
            CREATE INDEX IF NOT EXISTS idx_word_mastery_level ON word_mastery(mastery_level);

            CREATE TABLE IF NOT EXISTS pronunciation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                word TEXT,
                score INTEGER,
                attempt_time TIMESTAMP,
                recognized_text TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS conversation_practice (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                turn_number INTEGER DEFAULT 1,
                user_message TEXT,
                ai_response TEXT,
                corrections TEXT,
                suggestions TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                level INTEGER DEFAULT 1,
                xp INTEGER DEFAULT 0,
                total_xp INTEGER DEFAULT 0,
                total_study_time_minutes INTEGER DEFAULT 0,
                total_words_learned INTEGER DEFAULT 0,
                current_streak INTEGER DEFAULT 0,
                longest_streak INTEGER DEFAULT 0,
                last_study_date DATE,
                daily_goal_minutes INTEGER DEFAULT 15,
                best_quiz_score REAL DEFAULT 0,
                total_sessions INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS achievements (
                achievement_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                icon TEXT,
                category TEXT,
                requirement INTEGER,
                unlocked INTEGER DEFAULT 0,
                unlocked_at TIMESTAMP,
                progress INTEGER DEFAULT 0
            );
        ''')
        self.conn.commit()

    # ─── Session Management ───────────────────────────────────────────────────

    def start_session(self, lesson_number: int, session_type: str = 'lesson') -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO sessions (start_time, lesson_number, session_type) VALUES (?, ?, ?)',
            (datetime.now(), lesson_number, session_type)
        )
        self.conn.commit()
        return cursor.lastrowid

    def end_session(self, session_id: int, words_learned: int, quiz_score: Optional[float]):
        cursor = self.conn.cursor()
        cursor.execute(
            'UPDATE sessions SET end_time=?, words_learned=?, quiz_score=? WHERE id=?',
            (datetime.now(), words_learned, quiz_score, session_id)
        )
        # Update user_progress counters
        cursor.execute(
            'UPDATE user_progress SET total_sessions = total_sessions + 1 WHERE id = 1'
        )
        if quiz_score is not None:
            cursor.execute(
                'UPDATE user_progress SET best_quiz_score = MAX(best_quiz_score, ?) WHERE id = 1',
                (quiz_score,)
            )
        self.conn.commit()

    # ─── Word Mastery ─────────────────────────────────────────────────────────

    def update_word_mastery(
        self,
        word_data: Dict,
        is_correct: bool,
        easiness_factor: float = None,
        interval: int = None,
        repetitions: int = None,
        next_review: datetime = None,
        mastery_level: str = None,
    ):
        """단어 마스터 레벨 업데이트 (SM-2 필드 포함)"""
        cursor = self.conn.cursor()
        simplified = word_data['simplified']

        cursor.execute('SELECT times_practiced, times_correct FROM word_mastery WHERE simplified=?', (simplified,))
        result = cursor.fetchone()

        if result:
            tp = result[0] + 1
            tc = result[1] + (1 if is_correct else 0)
        else:
            tp = 1
            tc = 1 if is_correct else 0
            cursor.execute(
                'INSERT INTO word_mastery (simplified, traditional, pinyin, definitions) VALUES (?, ?, ?, ?)',
                (simplified, word_data.get('traditional', ''), word_data.get('pinyin', ''),
                 ', '.join(word_data.get('definitions', [])))
            )
            # Count toward total words
            cursor.execute(
                'UPDATE user_progress SET total_words_learned = total_words_learned + 1 WHERE id = 1'
            )

        # Calculate mastery if not provided
        if mastery_level is None:
            accuracy = tc / tp if tp > 0 else 0
            if tp >= 10 and accuracy >= 0.9:
                mastery_level = 'mastered'
                next_review_days = 30
            elif tp >= 5 and accuracy >= 0.7:
                mastery_level = 'proficient'
                next_review_days = 7
            elif tp >= 3:
                mastery_level = 'learning'
                next_review_days = 1
            else:
                mastery_level = 'new'
                next_review_days = 0
            if next_review is None:
                next_review = datetime.now() + timedelta(days=next_review_days)

        cursor.execute('''
            UPDATE word_mastery SET
                times_practiced=?, times_correct=?, last_practiced=?,
                mastery_level=?, next_review=?,
                easiness_factor=COALESCE(?, easiness_factor),
                interval=COALESCE(?, interval),
                repetitions=COALESCE(?, repetitions)
            WHERE simplified=?
        ''', (tp, tc, datetime.now(), mastery_level, next_review,
              easiness_factor, interval, repetitions, simplified))
        self.conn.commit()

    def get_word_data(self, simplified: str) -> Optional[Dict]:
        """단어 데이터 조회"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT word_id, simplified, traditional, pinyin, definitions,
                   times_practiced, times_correct, mastery_level,
                   easiness_factor, interval, repetitions, next_review
            FROM word_mastery WHERE simplified=?
        ''', (simplified,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "word_id": row[0], "simplified": row[1], "traditional": row[2],
            "pinyin": row[3], "definitions": row[4].split(", ") if row[4] else [],
            "times_practiced": row[5], "times_correct": row[6],
            "mastery_level": row[7], "easiness_factor": row[8],
            "interval": row[9], "repetitions": row[10], "next_review": row[11],
        }

    def get_words_for_review(self, limit: int = 20) -> List[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT simplified, mastery_level, pinyin, definitions
            FROM word_mastery
            WHERE next_review <= ? OR next_review IS NULL
            ORDER BY next_review ASC
            LIMIT ?
        ''', (datetime.now(), limit))
        return cursor.fetchall()

    # ─── Pronunciation ────────────────────────────────────────────────────────

    def record_pronunciation(self, word: str, score: int, recognized_text: str, session_id: int = None):
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO pronunciation_history (session_id, word, score, attempt_time, recognized_text) VALUES (?, ?, ?, ?, ?)',
            (session_id, word, score, datetime.now(), recognized_text)
        )
        self.conn.commit()

    # ─── Conversation ─────────────────────────────────────────────────────────

    def save_conversation_turn(self, session_id: int, turn: int, user_msg: str, ai_msg: str,
                                corrections: str = None, suggestions: str = None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO conversation_practice
            (session_id, turn_number, user_message, ai_response, corrections, suggestions, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, turn, user_msg, ai_msg, corrections, suggestions, datetime.now()))
        self.conn.commit()

    # ─── User Progress & Gamification ─────────────────────────────────────────

    def init_user_progress(self):
        """user_progress 싱글톤 레코드 초기화"""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT OR IGNORE INTO user_progress (id) VALUES (1)'
        )
        self.conn.commit()

    def get_user_progress(self) -> Dict:
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM user_progress WHERE id=1')
        row = cursor.fetchone()
        if not row:
            return {}
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, row))

    def add_xp(self, amount: int) -> int:
        """XP 추가 및 새 total_xp 반환"""
        cursor = self.conn.cursor()
        cursor.execute(
            'UPDATE user_progress SET xp=xp+?, total_xp=total_xp+? WHERE id=1',
            (amount, amount)
        )
        self.conn.commit()
        cursor.execute('SELECT total_xp FROM user_progress WHERE id=1')
        row = cursor.fetchone()
        return row[0] if row else 0

    def update_streak(self) -> Dict:
        """오늘 학습 처리 및 연속 학습 업데이트"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT current_streak, longest_streak, last_study_date FROM user_progress WHERE id=1')
        row = cursor.fetchone()
        if not row:
            return {}

        current_streak, longest_streak, last_study_date = row
        today = date.today()

        if last_study_date:
            last = date.fromisoformat(str(last_study_date))
            if last == today:
                return {"current_streak": current_streak, "already_done": True}
            elif last == today - timedelta(days=1):
                current_streak += 1
            else:
                current_streak = 1
        else:
            current_streak = 1

        longest_streak = max(longest_streak or 0, current_streak)
        cursor.execute('''
            UPDATE user_progress SET
                current_streak=?, longest_streak=?, last_study_date=?
            WHERE id=1
        ''', (current_streak, longest_streak, today.isoformat()))
        self.conn.commit()
        return {"current_streak": current_streak, "longest_streak": longest_streak, "already_done": False}

    # ─── Achievements ─────────────────────────────────────────────────────────

    def init_achievements(self, achievements: List[Dict]):
        """업적 초기화"""
        cursor = self.conn.cursor()
        for ach in achievements:
            cursor.execute('''
                INSERT OR IGNORE INTO achievements
                (achievement_id, name, description, icon, category, requirement)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (ach["id"], ach["name"], ach["description"],
                  ach.get("icon", ""), ach["category"], ach["requirement"]))
        self.conn.commit()

    def is_achievement_unlocked(self, achievement_id: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute('SELECT unlocked FROM achievements WHERE achievement_id=?', (achievement_id,))
        row = cursor.fetchone()
        return bool(row and row[0])

    def unlock_achievement(self, achievement_id: str):
        cursor = self.conn.cursor()
        cursor.execute(
            'UPDATE achievements SET unlocked=1, unlocked_at=? WHERE achievement_id=?',
            (datetime.now(), achievement_id)
        )
        self.conn.commit()

    def get_achievements(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('SELECT achievement_id, name, description, icon, category, unlocked, unlocked_at FROM achievements')
        rows = cursor.fetchall()
        return [
            {
                "id": r[0], "name": r[1], "description": r[2],
                "icon": r[3], "category": r[4],
                "unlocked": bool(r[5]), "unlocked_at": r[6],
            }
            for r in rows
        ]

    # ─── Statistics ───────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT SUM((julianday(end_time) - julianday(start_time)) * 24 * 60)
            FROM sessions WHERE end_time IS NOT NULL
        ''')
        total_minutes = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM word_mastery WHERE mastery_level='mastered'")
        mastered_words = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM word_mastery")
        total_words_learned = cursor.fetchone()[0]

        cursor.execute('SELECT AVG(quiz_score) FROM sessions WHERE quiz_score IS NOT NULL')
        avg_quiz_score = cursor.fetchone()[0] or 0

        cursor.execute('SELECT MAX(quiz_score) FROM sessions WHERE quiz_score IS NOT NULL')
        best_quiz_score = cursor.fetchone()[0] or 0

        cursor.execute('SELECT AVG(score) FROM pronunciation_history WHERE attempt_time >= date("now", "-7 days")')
        avg_pronunciation = cursor.fetchone()[0] or 0

        cursor.execute('SELECT COUNT(*) FROM sessions')
        total_sessions = cursor.fetchone()[0]

        up = self.get_user_progress()

        return {
            'total_study_hours': total_minutes / 60,
            'total_study_minutes': int(total_minutes),
            'mastered_words': mastered_words,
            'total_words_learned': total_words_learned,
            'average_quiz_score': round(avg_quiz_score, 1),
            'best_quiz_score': round(best_quiz_score, 1),
            'average_pronunciation': round(avg_pronunciation, 1),
            'total_sessions': total_sessions,
            'current_streak': up.get('current_streak', 0),
            'longest_streak': up.get('longest_streak', 0),
            'level': up.get('level', 1),
            'total_xp': up.get('total_xp', 0),
        }

    def get_learning_curve(self, days: int = 30) -> List[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DATE(start_time) as d, COUNT(*) as sessions, AVG(quiz_score) as avg_score
            FROM sessions
            WHERE start_time >= date('now', '-' || ? || ' days')
            GROUP BY DATE(start_time)
            ORDER BY d
        ''', (days,))
        return cursor.fetchall()

    def close(self):
        if self.conn:
            self.conn.close()
