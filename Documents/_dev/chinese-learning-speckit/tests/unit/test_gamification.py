"""
Unit tests for Gamification System
"""
import pytest
from src.learning.gamification import (
    calculate_level,
    xp_for_current_level,
    xp_for_next_level,
    xp_progress_in_level,
    XP_REWARDS,
    ACHIEVEMENTS,
)


class TestLevelCalculation:
    def test_level_1_at_zero_xp(self):
        assert calculate_level(0) == 1

    def test_level_1_below_threshold(self):
        assert calculate_level(50) == 1

    def test_level_2_at_100_xp(self):
        assert calculate_level(100) == 2

    def test_level_increases_monotonically(self):
        levels = [calculate_level(xp) for xp in [0, 50, 100, 200, 300, 500, 1000]]
        assert levels == sorted(levels)

    def test_high_xp_gives_high_level(self):
        assert calculate_level(20000) > 10

    def test_xp_for_next_level_positive(self):
        for level in range(1, 10):
            assert xp_for_next_level(level) > 0

    def test_xp_scaling(self):
        # Each level requires more XP than previous
        prev = xp_for_next_level(1)
        for level in range(2, 8):
            current = xp_for_next_level(level)
            assert current > prev
            prev = current

    def test_xp_progress_in_level_bounds(self):
        total_xp = 250
        current_in_level, level_total = xp_progress_in_level(total_xp)
        assert 0 <= current_in_level <= level_total


class TestXPRewards:
    def test_all_events_have_rewards(self):
        expected_events = [
            "word_learned", "word_mastered", "quiz_correct",
            "quiz_perfect", "conversation_turn", "daily_goal_met",
        ]
        for event in expected_events:
            assert event in XP_REWARDS
            assert XP_REWARDS[event] > 0

    def test_mastered_reward_greater_than_learned(self):
        assert XP_REWARDS["word_mastered"] > XP_REWARDS["word_learned"]

    def test_quiz_perfect_greater_than_correct(self):
        assert XP_REWARDS["quiz_perfect"] > XP_REWARDS["quiz_correct"]


class TestAchievements:
    def test_achievements_have_required_fields(self):
        for ach in ACHIEVEMENTS:
            assert "id" in ach
            assert "name" in ach
            assert "description" in ach
            assert "category" in ach
            assert "requirement" in ach

    def test_achievement_ids_unique(self):
        ids = [a["id"] for a in ACHIEVEMENTS]
        assert len(ids) == len(set(ids))

    def test_achievement_categories_valid(self):
        valid_cats = {"words", "streak", "score", "time", "special"}
        for ach in ACHIEVEMENTS:
            assert ach["category"] in valid_cats

    def test_achievements_have_icons(self):
        for ach in ACHIEVEMENTS:
            assert ach.get("icon"), f"Achievement {ach['id']} missing icon"
