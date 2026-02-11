"""
Multi-Agent System
Orchestrator + Content + Tutor + Eval + Data 에이전트 구현
"""

import os
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class AgentMessage:
    """에이전트 간 메시지 형식"""
    sender: str
    receiver: str
    task: str
    payload: dict
    priority: int = 1
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


def _get_client():
    """Anthropic 클라이언트 생성"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or not ANTHROPIC_AVAILABLE:
        return None
    return anthropic.Anthropic(api_key=api_key)


def _call_claude(client, system: str, user: str, max_tokens: int = 1024) -> Optional[str]:
    """Claude API 호출 헬퍼"""
    if client is None:
        return None
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
    except Exception as e:
        print(f"[Claude API Error] {e}")
        return None


class ContentAgent:
    """콘텐츠 생성 에이전트 - 레슨, 대화문, 예문 생성"""

    def __init__(self):
        self.client = _get_client()

    def generate_lesson(self, words: List[str], level: str = "HSK1", theme: str = None) -> dict:
        """
        단어 리스트로 학습 레슨 생성

        Args:
            words: 한자 단어 리스트
            level: HSK 레벨
            theme: 주제 (선택)

        Returns:
            {"dialogues": [...], "grammar_points": [...], "exercises": [...]}
        """
        theme_str = f"주제: {theme}" if theme else "일상 회화"
        prompt = f"""다음 중국어 단어들로 {level} 레벨 학습 자료를 만들어주세요.
단어: {', '.join(words)}
{theme_str}

JSON 형식으로 응답:
{{
  "dialogues": [
    {{
      "context": "상황 설명",
      "chinese": "중국어 대화",
      "pinyin": "병음",
      "korean": "한국어 번역"
    }}
  ],
  "grammar_points": ["문법 포인트 1", "문법 포인트 2"],
  "exercises": ["연습 문제 1", "연습 문제 2"]
}}"""

        system = "당신은 중국어 교육 전문가입니다. 한국어 학습자를 위한 HSK 학습 자료를 만듭니다."
        response = _call_claude(self.client, system, prompt)

        if response:
            try:
                # JSON 블록 추출
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception:
                pass

        # 폴백: 기본 템플릿
        return {
            "dialogues": [
                {
                    "context": "기본 인사",
                    "chinese": "你好！你叫什么名字？",
                    "pinyin": "Nǐ hǎo! Nǐ jiào shénme míngzi?",
                    "korean": "안녕하세요! 이름이 뭐예요?",
                }
            ],
            "grammar_points": ["叫 + 名字 (이름 표현)", "是 구문 (A는 B입니다)"],
            "exercises": ["빈칸 채우기: 我___学生。(是)", "번역: 안녕하세요 → ___"],
        }

    def generate_dialogue(self, vocabulary: List[str], situation: str, num_exchanges: int = 4) -> dict:
        """
        특정 상황에 맞는 대화문 생성

        Args:
            vocabulary: 사용할 단어 목록
            situation: 상황 설명 (한국어)
            num_exchanges: 대화 교환 횟수

        Returns:
            {"situation": str, "exchanges": [...], "grammar_notes": [...]}
        """
        prompt = f"""다음 조건으로 중국어 대화문을 만들어주세요:
- 상황: {situation}
- 사용 단어: {', '.join(vocabulary)}
- 대화 횟수: {num_exchanges}번 교환
- 난이도: HSK 1-2급

JSON 형식:
{{
  "situation": "상황 설명",
  "exchanges": [
    {{"speaker": "A", "chinese": "", "pinyin": "", "korean": ""}},
    {{"speaker": "B", "chinese": "", "pinyin": "", "korean": ""}}
  ],
  "grammar_notes": ["문법 노트"]
}}"""

        system = "당신은 중국어 교육 전문가입니다."
        response = _call_claude(self.client, system, prompt)

        if response:
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception:
                pass

        return {
            "situation": situation,
            "exchanges": [
                {"speaker": "A", "chinese": "你好！", "pinyin": "Nǐ hǎo!", "korean": "안녕하세요!"},
                {"speaker": "B", "chinese": "你好！", "pinyin": "Nǐ hǎo!", "korean": "안녕하세요!"},
            ],
            "grammar_notes": [],
        }


class TutorAgent:
    """개인 튜터 에이전트 - 대화, 문법 교정, 설명"""

    def __init__(self):
        self.client = _get_client()
        self.conversation_history: List[dict] = []

    def chat(self, user_message: str, user_level: str = "beginner") -> dict:
        """
        사용자 메시지에 응답 생성 (교정 포함)

        Args:
            user_message: 사용자 메시지
            user_level: 학습 수준

        Returns:
            {
                "response": str,
                "response_pinyin": str,
                "response_translation": str,
                "corrections": [...],
                "suggestions": [...],
                "encouragement": str
            }
        """
        system = f"""당신은 친절한 중국어 튜터입니다. 한국어를 사용하는 {user_level} 수준의 학습자와 대화합니다.

역할:
1. 중국어로 자연스럽게 대화 (HSK 1-2급 수준)
2. 문법 오류 교정
3. 격려와 피드백 제공

응답 형식 (JSON):
{{
  "response": "중국어 응답",
  "response_pinyin": "병음",
  "response_translation": "한국어 번역",
  "corrections": [
    {{"original": "틀린 표현", "corrected": "올바른 표현", "explanation": "설명"}}
  ],
  "suggestions": ["개선 제안"],
  "encouragement": "격려 메시지"
}}"""

        self.conversation_history.append({"role": "user", "content": user_message})

        if self.client:
            try:
                messages = self.conversation_history[-10:]  # 최근 10개만
                response = self.client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                )
                text = response.content[0].text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": result.get("response", ""),
                    })
                    return result
            except Exception as e:
                print(f"[TutorAgent Error] {e}")

        # 폴백 응답
        fallback = {
            "response": "很好！继续加油！",
            "response_pinyin": "Hěn hǎo! Jìxù jiāyóu!",
            "response_translation": "매우 좋아요! 계속 화이팅!",
            "corrections": [],
            "suggestions": ["더 많은 단어를 사용해보세요!"],
            "encouragement": "잘 하고 있어요! 계속 연습하세요!",
        }
        self.conversation_history.append({"role": "assistant", "content": fallback["response"]})
        return fallback

    def explain_grammar(self, sentence: str) -> dict:
        """
        문장의 문법 설명 생성

        Args:
            sentence: 분석할 중국어 문장

        Returns:
            {"grammar_points": [...], "similar_examples": [...], "tips": [...]}
        """
        prompt = f"""다음 중국어 문장을 한국어 학습자를 위해 설명해주세요: {sentence}

JSON 형식:
{{
  "sentence": "{sentence}",
  "structure": "문장 구조 설명",
  "grammar_points": [
    {{"point": "문법 요소", "explanation": "설명", "example": "예문"}}
  ],
  "similar_examples": ["비슷한 예문 1", "비슷한 예문 2"],
  "tips": ["학습 팁 1"]
}}"""

        system = "중국어 문법 전문가입니다. 한국어로 설명해주세요."
        response = _call_claude(self.client, system, prompt)

        if response:
            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception:
                pass

        return {
            "sentence": sentence,
            "structure": "주어 + 동사 + 목적어",
            "grammar_points": [],
            "similar_examples": [],
            "tips": ["기본 문형을 먼저 익히세요"],
        }

    def reset_conversation(self):
        """대화 히스토리 초기화"""
        self.conversation_history = []


class EvalAgent:
    """평가 에이전트 - 퀴즈 생성, 채점, 약점 분석"""

    def __init__(self):
        self.client = _get_client()

    def generate_adaptive_quiz(self, words: List[dict], recent_scores: List[float], count: int = 10) -> List[dict]:
        """
        적응형 퀴즈 생성 (성적 기반 난이도 조절)

        Args:
            words: 단어 목록 (dict with simplified, pinyin, definitions)
            recent_scores: 최근 퀴즈 점수 목록
            count: 문제 수

        Returns:
            퀴즈 문제 목록
        """
        difficulty = self._calculate_difficulty(recent_scores)
        return self._generate_questions(words, difficulty, count)

    def _calculate_difficulty(self, scores: List[float]) -> str:
        """성적 기반 난이도 계산"""
        if not scores:
            return "medium"
        avg = sum(scores) / len(scores)
        if avg >= 90:
            return "hard"
        elif avg >= 70:
            return "medium"
        return "easy"

    def _generate_questions(self, words: List[dict], difficulty: str, count: int) -> List[dict]:
        """퀴즈 문제 생성"""
        import random
        questions = []
        available = words[:count] if len(words) >= count else words

        for i, word in enumerate(available):
            q_type = random.choice(["translation", "fill_blank", "pinyin"])

            if q_type == "translation":
                # 중국어 → 한국어 선택
                defs = word.get("definitions", ["모름"])
                correct = defs[0] if defs else "모름"
                wrong_words = [w for w in words if w != word]
                random.shuffle(wrong_words)
                wrong_answers = [w.get("definitions", ["?"])[0] for w in wrong_words[:3]]
                options = [correct] + wrong_answers
                random.shuffle(options)

                questions.append({
                    "id": f"q_{i}",
                    "type": "translation",
                    "question": f"'{word['simplified']}' 의 뜻은?",
                    "options": options,
                    "answer": correct,
                    "explanation": f"{word['simplified']} ({word.get('pinyin', '')}) = {correct}",
                    "points": 10 if difficulty == "hard" else 5,
                })

            elif q_type == "fill_blank":
                defs = word.get("definitions", ["모름"])
                meaning = defs[0] if defs else "모름"
                questions.append({
                    "id": f"q_{i}",
                    "type": "fill_blank",
                    "question": f"'{meaning}'를 중국어로? (병음: {word.get('pinyin', '')})",
                    "options": [],
                    "answer": word["simplified"],
                    "explanation": f"{meaning} = {word['simplified']}",
                    "points": 10,
                })

            elif q_type == "pinyin":
                pinyin = word.get("pinyin", "")
                wrong_words = [w for w in words if w != word]
                random.shuffle(wrong_words)
                wrong_pinyins = [w.get("pinyin", "?") for w in wrong_words[:3]]
                options = [pinyin] + wrong_pinyins
                random.shuffle(options)

                questions.append({
                    "id": f"q_{i}",
                    "type": "pinyin",
                    "question": f"'{word['simplified']}' 의 병음은?",
                    "options": options,
                    "answer": pinyin,
                    "explanation": f"{word['simplified']} = {pinyin}",
                    "points": 5,
                })

        return questions[:count]

    def evaluate_answer(self, question: dict, user_answer: str) -> dict:
        """
        답안 평가

        Args:
            question: 퀴즈 문제 dict
            user_answer: 사용자 답안

        Returns:
            {"is_correct": bool, "score": int, "feedback": str}
        """
        correct = question.get("answer", "")
        is_correct = user_answer.strip() == correct.strip()

        return {
            "is_correct": is_correct,
            "score": question.get("points", 5) if is_correct else 0,
            "feedback": "정답입니다! 🎉" if is_correct else f"오답. 정답: {correct}",
            "explanation": question.get("explanation", ""),
        }


class DataAgent:
    """데이터 분석 에이전트 - 진도 분석, 인사이트, 추천"""

    def __init__(self, progress_tracker):
        self.tracker = progress_tracker

    def analyze_progress(self, period: str = "week") -> dict:
        """
        학습 진도 분석

        Args:
            period: "day", "week", "month", "all"

        Returns:
            {"summary": dict, "insights": [...], "recommendations": [...]}
        """
        stats = self.tracker.get_statistics()
        level_data = self.tracker.get_user_progress()

        insights = []
        recommendations = []

        # 인사이트 생성
        mastered = stats.get("mastered_words", 0)
        total = stats.get("total_words_learned", 0)
        if total > 0:
            mastery_rate = mastered / total
            if mastery_rate < 0.3:
                insights.append("⚠️ 마스터한 단어 비율이 낮습니다. 복습을 늘려보세요!")
                recommendations.append("매일 5분 복습 세션 추가")
            elif mastery_rate > 0.7:
                insights.append("✨ 마스터 비율이 높습니다! 새 단어에 도전해보세요.")
                recommendations.append("다음 레슨으로 진행하세요")

        streak = level_data.get("current_streak", 0)
        if streak == 0:
            recommendations.append("오늘 학습을 시작해서 연속 학습을 시작하세요!")
        elif streak >= 7:
            insights.append(f"🔥 {streak}일 연속 학습 중! 대단해요!")

        avg_score = stats.get("average_quiz_score", 0)
        if avg_score > 0:
            if avg_score < 60:
                insights.append("📉 퀴즈 점수가 낮습니다. 단어 복습이 필요합니다.")
                recommendations.append("퀴즈 전에 단어 카드를 복습하세요")
            elif avg_score >= 90:
                insights.append("🎯 퀴즈 성적이 우수합니다!")

        return {
            "summary": {
                "total_words": total,
                "mastered_words": mastered,
                "avg_quiz_score": avg_score,
                "current_streak": streak,
            },
            "insights": insights or ["아직 데이터가 충분하지 않습니다. 학습을 시작해보세요!"],
            "recommendations": recommendations or ["오늘 단어 10개를 학습해보세요!"],
        }

    def predict_retention(self, word_list: List[str]) -> List[dict]:
        """
        단어 기억 유지율 예측 (간단한 규칙 기반)

        Args:
            word_list: 단어 목록

        Returns:
            단어별 유지율 예측
        """
        predictions = []
        for word in word_list:
            data = self.tracker.get_word_data(word)
            if not data:
                continue

            interval = data.get("interval", 0)
            repetitions = data.get("repetitions", 0)
            ef = data.get("easiness_factor", 2.5)

            # 단순 추정: 반복 횟수와 EF 기반
            retention = min(0.95, 0.5 + (repetitions * 0.1) + (ef - 1.3) * 0.1)

            predictions.append({
                "word": word,
                "retention_probability": round(retention, 2),
                "days_since_review": interval,
                "recommended_action": "복습 필요" if retention < 0.7 else "OK",
            })

        return predictions


class OrchestratorAgent:
    """오케스트레이터 에이전트 - 사용자 의도 파악 및 전문 에이전트 조율"""

    def __init__(self, progress_tracker=None):
        self.content_agent = ContentAgent()
        self.tutor_agent = TutorAgent()
        self.eval_agent = EvalAgent()
        self.data_agent = DataAgent(progress_tracker) if progress_tracker else None

    def process_query(self, query: str, context: dict = None) -> dict:
        """
        사용자 쿼리 분석 및 적절한 에이전트에 위임

        Args:
            query: 사용자 입력
            context: 현재 학습 컨텍스트

        Returns:
            {"intent": str, "result": dict, "agent_used": str}
        """
        if context is None:
            context = {}

        intent = self._classify_intent(query)

        if intent == "learn":
            words = context.get("current_words", ["你好", "谢谢", "再见"])
            result = self.content_agent.generate_lesson(words)
            return {"intent": intent, "result": result, "agent_used": "ContentAgent"}

        elif intent == "practice":
            result = self.tutor_agent.chat(query, context.get("user_level", "beginner"))
            return {"intent": intent, "result": result, "agent_used": "TutorAgent"}

        elif intent == "quiz":
            words = context.get("vocabulary", [])
            scores = context.get("recent_scores", [])
            result = self.eval_agent.generate_adaptive_quiz(words, scores)
            return {"intent": intent, "result": result, "agent_used": "EvalAgent"}

        elif intent == "progress" and self.data_agent:
            result = self.data_agent.analyze_progress()
            return {"intent": intent, "result": result, "agent_used": "DataAgent"}

        else:
            result = self.tutor_agent.chat(query)
            return {"intent": "general", "result": result, "agent_used": "TutorAgent"}

    def _classify_intent(self, query: str) -> str:
        """
        사용자 쿼리에서 의도 분류

        Returns:
            "learn", "practice", "quiz", "progress", "general"
        """
        query_lower = query.lower()

        learn_keywords = ["배우", "학습", "단어", "lesson", "배울", "learn"]
        practice_keywords = ["연습", "대화", "practice", "chat", "얘기", "말하기"]
        quiz_keywords = ["퀴즈", "시험", "quiz", "test", "문제", "채점"]
        progress_keywords = ["진도", "통계", "progress", "stats", "성적", "얼마나"]

        if any(k in query_lower for k in learn_keywords):
            return "learn"
        elif any(k in query_lower for k in practice_keywords):
            return "practice"
        elif any(k in query_lower for k in quiz_keywords):
            return "quiz"
        elif any(k in query_lower for k in progress_keywords):
            return "progress"
        return "general"
