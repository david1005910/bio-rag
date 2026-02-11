"""
Unit tests for ProgressTracker (database operations)
"""
import pytest
import os
import tempfile
from datetime import datetime

from src.core.progress_tracker import ProgressTracker


@pytest.fixture
def tracker(tmp_path):
    db_path = str(tmp_path / "test.db")
    t = ProgressTracker(db_path)
    t.init_user_progress()
    yield t
    t.close()


class TestSessionManagement:
    def test_start_and_end_session(self, tracker):
        session_id = tracker.start_session(1)
        assert session_id > 0
        tracker.end_session(session_id, words_learned=5, quiz_score=80.0)
        stats = tracker.get_statistics()
        assert stats["total_sessions"] >= 1

    def test_multiple_sessions(self, tracker):
        for i in range(3):
            sid = tracker.start_session(i)
            tracker.end_session(sid, 5, float(70 + i * 10))
        stats = tracker.get_statistics()
        assert stats["total_sessions"] == 3
        assert stats["average_quiz_score"] == 80.0


class TestWordMastery:
    def test_add_new_word(self, tracker):
        word = {"simplified": "你好", "traditional": "你好", "pinyin": "nǐ hǎo",
                "definitions": ["안녕하세요"]}
        tracker.update_word_mastery(word, True)
        data = tracker.get_word_data("你好")
        assert data is not None
        assert data["simplified"] == "你好"
        assert data["times_practiced"] == 1
        assert data["times_correct"] == 1

    def test_word_mastery_increments(self, tracker):
        word = {"simplified": "谢谢", "pinyin": "xiè xiè", "definitions": ["감사합니다"]}
        tracker.update_word_mastery(word, True)
        tracker.update_word_mastery(word, False)
        data = tracker.get_word_data("谢谢")
        assert data["times_practiced"] == 2
        assert data["times_correct"] == 1

    def test_words_for_review(self, tracker):
        word = {"simplified": "再见", "pinyin": "zài jiàn", "definitions": ["안녕히"]}
        tracker.update_word_mastery(word, True)
        # After initial add, next_review is None or in past → should appear
        reviews = tracker.get_words_for_review(10)
        assert any(r[0] == "再见" for r in reviews)


class TestUserProgress:
    def test_initial_xp_zero(self, tracker):
        progress = tracker.get_user_progress()
        assert progress.get("total_xp", 0) == 0

    def test_add_xp(self, tracker):
        new_total = tracker.add_xp(100)
        assert new_total == 100
        new_total = tracker.add_xp(50)
        assert new_total == 150

    def test_streak_first_day(self, tracker):
        result = tracker.update_streak()
        assert result["current_streak"] == 1
        assert not result.get("already_done")

    def test_streak_same_day_no_increment(self, tracker):
        tracker.update_streak()
        result = tracker.update_streak()
        assert result.get("already_done")


class TestAchievements:
    def test_init_achievements(self, tracker):
        achievements = [
            {"id": "test1", "name": "Test 1", "description": "Desc", "icon": "🏅",
             "category": "words", "requirement": 1},
        ]
        tracker.init_achievements(achievements)
        all_ach = tracker.get_achievements()
        assert any(a["id"] == "test1" for a in all_ach)

    def test_unlock_achievement(self, tracker):
        achievements = [
            {"id": "unlock_test", "name": "Unlock", "description": "Test",
             "icon": "🏆", "category": "words", "requirement": 1}
        ]
        tracker.init_achievements(achievements)
        assert not tracker.is_achievement_unlocked("unlock_test")
        tracker.unlock_achievement("unlock_test")
        assert tracker.is_achievement_unlocked("unlock_test")

    def test_double_init_no_error(self, tracker):
        achievements = [
            {"id": "dup_test", "name": "Dup", "description": "Dup", "icon": "🏅",
             "category": "words", "requirement": 1}
        ]
        tracker.init_achievements(achievements)
        tracker.init_achievements(achievements)  # Should not raise


class TestStatistics:
    def test_empty_stats(self, tracker):
        stats = tracker.get_statistics()
        assert stats["total_study_hours"] == 0
        assert stats["mastered_words"] == 0
        assert stats["average_quiz_score"] == 0

    def test_stats_after_session(self, tracker):
        sid = tracker.start_session(0)
        tracker.end_session(sid, 10, 90.0)
        stats = tracker.get_statistics()
        assert stats["total_sessions"] == 1
        assert stats["average_quiz_score"] == 90.0
        assert stats["best_quiz_score"] == 90.0
