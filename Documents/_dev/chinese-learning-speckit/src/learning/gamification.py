"""
Gamification System
레벨, XP, 업적, 연속 학습 관리
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple
import json


# XP 보상 정의
XP_REWARDS = {
    "word_learned": 10,
    "word_mastered": 50,
    "quiz_correct": 5,
    "quiz_perfect": 30,       # 100% 퀴즈
    "conversation_turn": 8,
    "pronunciation_good": 15,  # 80점 이상
    "daily_goal_met": 20,
    "streak_bonus_7": 50,
    "streak_bonus_30": 200,
}

# 업적 정의
ACHIEVEMENTS = [
    # 단어 업적
    {"id": "first_word", "name": "첫 걸음", "description": "첫 단어를 배웠습니다!", "category": "words", "requirement": 1, "icon": "🌱"},
    {"id": "words_10", "name": "초보 학습자", "description": "10개 단어를 배웠습니다!", "category": "words", "requirement": 10, "icon": "📖"},
    {"id": "words_50", "name": "중급 학습자", "description": "50개 단어를 배웠습니다!", "category": "words", "requirement": 50, "icon": "📚"},
    {"id": "words_100", "name": "단어 수집가", "description": "100개 단어를 배웠습니다!", "category": "words", "requirement": 100, "icon": "🏆"},

    # 연속 학습 업적
    {"id": "streak_3", "name": "꾸준한 학습자", "description": "3일 연속 학습!", "category": "streak", "requirement": 3, "icon": "🔥"},
    {"id": "streak_7", "name": "일주일 챌린지", "description": "7일 연속 학습!", "category": "streak", "requirement": 7, "icon": "⚡"},
    {"id": "streak_30", "name": "학습 마스터", "description": "30일 연속 학습!", "category": "streak", "requirement": 30, "icon": "👑"},

    # 퀴즈 업적
    {"id": "quiz_first", "name": "첫 퀴즈", "description": "첫 퀴즈를 완료했습니다!", "category": "score", "requirement": 1, "icon": "✏️"},
    {"id": "quiz_perfect", "name": "완벽한 점수", "description": "퀴즈에서 100점을 받았습니다!", "category": "score", "requirement": 100, "icon": "💯"},
    {"id": "quiz_10", "name": "퀴즈 마니아", "description": "10번 퀴즈를 완료했습니다!", "category": "score", "requirement": 10, "icon": "🎯"},

    # 시간 업적
    {"id": "time_1h", "name": "1시간 학습", "description": "총 1시간 학습했습니다!", "category": "time", "requirement": 60, "icon": "⏱️"},
    {"id": "time_10h", "name": "10시간 학습", "description": "총 10시간 학습했습니다!", "category": "time", "requirement": 600, "icon": "📅"},

    # 특별 업적
    {"id": "hsk1_complete", "name": "HSK 1급 완성", "description": "HSK 1급 모든 단어를 마스터!", "category": "special", "requirement": 150, "icon": "🎓"},
]


def calculate_level(total_xp: int) -> int:
    """
    총 XP에서 현재 레벨 계산
    Formula: XP(n) = 100 × 1.5^(n-1)

    Level 1: 0-100 XP
    Level 2: 100-250 XP
    Level 3: 250-475 XP
    ...
    """
    if total_xp <= 0:
        return 1

    level = 1
    xp_accumulated = 0

    while True:
        xp_for_this_level = int(100 * (1.5 ** (level - 1)))
        if xp_accumulated + xp_for_this_level > total_xp:
            break
        xp_accumulated += xp_for_this_level
        level += 1

    return level


def xp_for_current_level(level: int) -> int:
    """현재 레벨 시작까지 필요한 총 XP"""
    return sum(int(100 * (1.5 ** (l - 1))) for l in range(1, level))


def xp_for_next_level(level: int) -> int:
    """이 레벨에서 다음 레벨까지 필요한 XP"""
    return int(100 * (1.5 ** (level - 1)))


def xp_progress_in_level(total_xp: int) -> Tuple[int, int]:
    """
    현재 레벨 내 진행도 계산

    Returns:
        (현재 레벨 내 XP, 이 레벨 총 필요 XP)
    """
    level = calculate_level(total_xp)
    level_start_xp = xp_for_current_level(level)
    level_total = xp_for_next_level(level)
    current_in_level = total_xp - level_start_xp
    return current_in_level, level_total


class GamificationSystem:
    """게임화 시스템 관리자"""

    def __init__(self, progress_tracker):
        """
        Args:
            progress_tracker: ProgressTracker 인스턴스
        """
        self.tracker = progress_tracker
        self._ensure_user_progress()
        self._ensure_achievements()

    def _ensure_user_progress(self):
        """user_progress 테이블에 초기 레코드 생성"""
        self.tracker.init_user_progress()

    def _ensure_achievements(self):
        """achievements 테이블에 모든 업적 초기화"""
        self.tracker.init_achievements(ACHIEVEMENTS)

    def award_xp(self, event: str, extra_multiplier: float = 1.0) -> dict:
        """
        이벤트에 따른 XP 지급

        Args:
            event: XP_REWARDS 키 중 하나
            extra_multiplier: 보너스 배율

        Returns:
            {"xp_gained": int, "total_xp": int, "level": int, "leveled_up": bool}
        """
        xp_amount = int(XP_REWARDS.get(event, 0) * extra_multiplier)
        if xp_amount <= 0:
            return {"xp_gained": 0, "total_xp": 0, "level": 1, "leveled_up": False}

        old_progress = self.tracker.get_user_progress()
        old_level = calculate_level(old_progress.get("total_xp", 0))

        new_total = self.tracker.add_xp(xp_amount)
        new_level = calculate_level(new_total)

        leveled_up = new_level > old_level

        return {
            "xp_gained": xp_amount,
            "total_xp": new_total,
            "level": new_level,
            "leveled_up": leveled_up,
        }

    def update_streak(self) -> dict:
        """
        오늘 학습 처리 및 연속 학습 업데이트

        Returns:
            streak 정보 dict
        """
        return self.tracker.update_streak()

    def check_achievements(self) -> List[dict]:
        """
        새로 달성한 업적 확인 및 반환

        Returns:
            새로 잠금 해제된 업적 목록
        """
        stats = self.tracker.get_statistics()
        progress = self.tracker.get_user_progress()
        newly_unlocked = []

        for ach in ACHIEVEMENTS:
            if self.tracker.is_achievement_unlocked(ach["id"]):
                continue

            unlocked = False
            req = ach["requirement"]
            category = ach["category"]

            if category == "words":
                unlocked = stats.get("total_words_learned", 0) >= req
            elif category == "streak":
                unlocked = progress.get("current_streak", 0) >= req
            elif category == "score":
                if ach["id"] == "quiz_first":
                    unlocked = stats.get("total_sessions", 0) >= 1
                elif ach["id"] == "quiz_perfect":
                    unlocked = stats.get("best_quiz_score", 0) >= 100
                elif ach["id"] == "quiz_10":
                    unlocked = stats.get("total_sessions", 0) >= req
            elif category == "time":
                unlocked = stats.get("total_study_minutes", 0) >= req
            elif category == "special":
                unlocked = stats.get("mastered_words", 0) >= req

            if unlocked:
                self.tracker.unlock_achievement(ach["id"])
                newly_unlocked.append(ach)
                # 업적 달성 XP 보상
                self.award_xp("word_mastered")

        return newly_unlocked

    def get_level_info(self) -> dict:
        """현재 레벨 정보 반환"""
        progress = self.tracker.get_user_progress()
        total_xp = progress.get("total_xp", 0)
        level = calculate_level(total_xp)
        current_in_level, level_total = xp_progress_in_level(total_xp)

        return {
            "level": level,
            "total_xp": total_xp,
            "current_in_level": current_in_level,
            "xp_for_next_level": level_total,
            "progress_percent": min(100, int(current_in_level / level_total * 100)),
            "current_streak": progress.get("current_streak", 0),
            "longest_streak": progress.get("longest_streak", 0),
        }

    def get_all_achievements(self) -> List[dict]:
        """전체 업적 목록 및 잠금 상태 반환"""
        return self.tracker.get_achievements()
