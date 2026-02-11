"""
SM-2 Spaced Repetition System
간격 반복 학습 시스템 구현 (SuperMemo 2 알고리즘)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class Card:
    """플래시카드 데이터 클래스"""
    word_id: int
    simplified: str
    pinyin: str
    definitions: List[str]

    # SM-2 필드
    easiness_factor: float = 2.5
    interval: int = 0        # days
    repetitions: int = 0
    next_review: Optional[datetime] = None

    # 통계
    times_practiced: int = 0
    times_correct: int = 0
    mastery_level: str = "new"  # new, learning, proficient, mastered

    def __post_init__(self):
        if self.next_review is None:
            self.next_review = datetime.now()


def sm2_algorithm(card: Card, quality: int) -> Card:
    """
    SuperMemo 2 알고리즘으로 다음 복습 일정 계산

    Quality Scale:
        5: 완벽한 응답
        4: 약간의 망설임 후 정답
        3: 어렵게 정답
        2: 오답이지만 기억함
        1: 오답, 친숙한 느낌
        0: 완전히 잊음

    Args:
        card: 플래시카드 데이터
        quality: 응답 품질 (0-5)

    Returns:
        업데이트된 카드 객체
    """
    if not 0 <= quality <= 5:
        raise ValueError(f"Quality must be 0-5, got {quality}")

    card.times_practiced += 1
    if quality >= 3:
        card.times_correct += 1

    if quality < 3:
        # 기억 실패 시 초기화
        card.repetitions = 0
        card.interval = 1
    else:
        # 반복 횟수에 따른 간격 계산
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = round(card.interval * card.easiness_factor)

        card.repetitions += 1

    # Easiness Factor 업데이트 (최소 1.3)
    card.easiness_factor = max(
        1.3,
        card.easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    )

    # 다음 복습 날짜 설정
    card.next_review = datetime.now() + timedelta(days=card.interval)

    # 마스터리 레벨 업데이트
    card.mastery_level = _calculate_mastery_level(card)

    return card


def _calculate_mastery_level(card: Card) -> str:
    """
    카드의 마스터리 레벨 계산

    Returns:
        "new", "learning", "proficient", "mastered"
    """
    if card.times_practiced == 0:
        return "new"

    accuracy = card.times_correct / card.times_practiced if card.times_practiced > 0 else 0

    if card.repetitions >= 5 and accuracy >= 0.9 and card.interval >= 21:
        return "mastered"
    elif card.repetitions >= 3 and accuracy >= 0.75:
        return "proficient"
    elif card.repetitions >= 1:
        return "learning"
    else:
        return "new"


class SpacedRepetitionSystem:
    """간격 반복 시스템 관리자"""

    def __init__(self, progress_tracker):
        """
        Args:
            progress_tracker: ProgressTracker 인스턴스 (DB 접근)
        """
        self.tracker = progress_tracker

    def get_due_cards(self, limit: int = 20) -> List[dict]:
        """
        오늘 복습해야 할 단어 목록 반환

        Args:
            limit: 최대 반환 개수

        Returns:
            복습 단어 리스트 (dict 형태)
        """
        words = self.tracker.get_words_for_review(limit)
        return [
            {
                "simplified": w[0],
                "pinyin": w[2],
                "definitions": w[3].split(", ") if w[3] else [],
                "mastery_level": w[1],
            }
            for w in words
        ]

    def process_review(self, word: str, quality: int) -> dict:
        """
        복습 결과 처리 및 DB 업데이트

        Args:
            word: 한자 (simplified)
            quality: 응답 품질 (0-5)

        Returns:
            업데이트 결과 dict
        """
        # DB에서 현재 카드 상태 로드
        card_data = self.tracker.get_word_data(word)
        if not card_data:
            return {"error": f"Word '{word}' not found"}

        card = Card(
            word_id=card_data.get("word_id", 0),
            simplified=word,
            pinyin=card_data.get("pinyin", ""),
            definitions=card_data.get("definitions", []),
            easiness_factor=card_data.get("easiness_factor", 2.5),
            interval=card_data.get("interval", 0),
            repetitions=card_data.get("repetitions", 0),
            times_practiced=card_data.get("times_practiced", 0),
            times_correct=card_data.get("times_correct", 0),
        )

        # SM-2 알고리즘 적용
        updated_card = sm2_algorithm(card, quality)

        # DB 업데이트
        is_correct = quality >= 3
        self.tracker.update_word_mastery(
            {"simplified": word},
            is_correct,
            easiness_factor=updated_card.easiness_factor,
            interval=updated_card.interval,
            repetitions=updated_card.repetitions,
            next_review=updated_card.next_review,
            mastery_level=updated_card.mastery_level,
        )

        return {
            "word": word,
            "quality": quality,
            "new_interval": updated_card.interval,
            "next_review": updated_card.next_review.strftime("%Y-%m-%d"),
            "mastery_level": updated_card.mastery_level,
            "easiness_factor": round(updated_card.easiness_factor, 2),
        }

    def get_statistics(self) -> dict:
        """
        SRS 전체 통계 반환

        Returns:
            dict with counts by mastery level and due today
        """
        stats = self.tracker.get_statistics()
        return {
            "total_words": stats.get("total_words", 0),
            "mastered": stats.get("mastered_words", 0),
            "due_today": len(self.get_due_cards(100)),
        }
