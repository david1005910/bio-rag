"""
Unit tests for SM-2 Spaced Repetition System
"""
import pytest
from datetime import datetime, timedelta
from src.learning.spaced_repetition import Card, sm2_algorithm, _calculate_mastery_level


def make_card(**kwargs):
    defaults = dict(word_id=1, simplified="你好", pinyin="nǐ hǎo", definitions=["안녕하세요"])
    defaults.update(kwargs)
    return Card(**defaults)


class TestSM2Algorithm:
    def test_perfect_response_increases_interval(self):
        card = make_card()
        card = sm2_algorithm(card, quality=5)
        assert card.interval == 1
        assert card.repetitions == 1

        card = sm2_algorithm(card, quality=5)
        assert card.interval == 6
        assert card.repetitions == 2

        card = sm2_algorithm(card, quality=5)
        assert card.interval > 6
        assert card.repetitions == 3

    def test_poor_response_resets_card(self):
        card = make_card(repetitions=5, interval=30, easiness_factor=2.5)
        card = sm2_algorithm(card, quality=2)
        assert card.repetitions == 0
        assert card.interval == 1

    def test_easiness_factor_minimum(self):
        card = make_card()
        for _ in range(20):
            card = sm2_algorithm(card, quality=0)
        assert card.easiness_factor >= 1.3

    def test_next_review_set(self):
        card = make_card()
        before = datetime.now()
        card = sm2_algorithm(card, quality=5)
        assert card.next_review > before

    def test_invalid_quality_raises(self):
        card = make_card()
        with pytest.raises(ValueError):
            sm2_algorithm(card, quality=6)
        with pytest.raises(ValueError):
            sm2_algorithm(card, quality=-1)

    def test_times_practiced_increments(self):
        card = make_card(times_practiced=3, times_correct=2)
        card = sm2_algorithm(card, quality=5)
        assert card.times_practiced == 4
        assert card.times_correct == 3

    def test_times_correct_not_incremented_on_failure(self):
        card = make_card(times_practiced=3, times_correct=2)
        card = sm2_algorithm(card, quality=1)
        assert card.times_practiced == 4
        assert card.times_correct == 2


class TestMasteryLevel:
    def test_new_card_is_new(self):
        card = make_card(times_practiced=0, times_correct=0, repetitions=0, interval=0)
        assert _calculate_mastery_level(card) == "new"

    def test_proficient_after_practice(self):
        card = make_card(times_practiced=5, times_correct=4, repetitions=3, interval=7)
        assert _calculate_mastery_level(card) == "proficient"

    def test_mastered_after_long_interval(self):
        card = make_card(times_practiced=12, times_correct=11, repetitions=6, interval=25)
        assert _calculate_mastery_level(card) == "mastered"

    def test_learning_level(self):
        card = make_card(times_practiced=3, times_correct=2, repetitions=1, interval=1)
        assert _calculate_mastery_level(card) == "learning"
