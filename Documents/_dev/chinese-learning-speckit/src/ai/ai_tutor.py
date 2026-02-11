"""
AI 튜터 모듈
Claude API를 활용한 대화 생성 및 학습 지원
"""

import os
from typing import List, Dict, Optional
try:
    import anthropic
except ImportError:
    anthropic = None
    print("Warning: anthropic package not installed. AI features will be limited.")


class ChineseAITutor:
    """중국어 AI 튜터"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Anthropic API 키 (없으면 환경변수에서 로드)
        """
        if anthropic is None:
            self.client = None
            print("AI Tutor disabled: anthropic package not available")
            return
            
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.conversation_history = []
    
    def generate_dialogue(self, 
                         vocabulary: List[str], 
                         level: str = "beginner",
                         situation: str = "일상") -> List[Dict]:
        """
        학습 단어를 활용한 실용적인 대화 생성
        
        Args:
            vocabulary: 학습할 단어 리스트
            level: 난이도 (beginner, intermediate, advanced)
            situation: 상황 (일상, 식당, 쇼핑 등)
            
        Returns:
            대화 리스트
        """
        if self.client is None:
            return self._generate_simple_dialogue(vocabulary)
        
        prompt = f"""
당신은 중국어 교육 전문가입니다. 다음 단어들을 사용하여 초보자를 위한 실용적인 대화를 생성해주세요.

단어 목록: {', '.join(vocabulary)}
난이도: {level}
상황: {situation}

다음 형식으로 5개의 대화를 만들어주세요:
1. 상황 설명 (한국어)
2. 중국어 문장
3. 병음 (성조 표시)
4. 한국어 번역
5. 문법 포인트 설명

실제 생활에서 자주 사용하는 표현으로 만들어주세요.
"""
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return self._parse_dialogue_response(message.content[0].text)
        except Exception as e:
            print(f"API Error: {e}")
            return self._generate_simple_dialogue(vocabulary)
    
    def chat_practice(self, user_input: str, context: str = "") -> Dict:
        """
        대화 연습 - 사용자 입력에 대한 응답 및 피드백
        
        Args:
            user_input: 사용자 입력 (중국어)
            context: 대화 컨텍스트
            
        Returns:
            응답 및 피드백 딕셔너리
        """
        if self.client is None:
            return {
                "response": "AI 기능이 비활성화되어 있습니다.",
                "suggestions": []
            }
        
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        system_prompt = f"""
당신은 친절한 중국어 회화 선생님입니다. 
학생과 중국어로 대화하면서 자연스럽게 교정해주세요.

규칙:
1. 학생의 중국어에 문법 오류가 있으면 정중하게 고쳐주세요
2. 더 자연스러운 표현을 제안해주세요
3. 간단한 중국어로 대화하되, 병음과 한국어 번역을 함께 제공하세요
4. 격려하고 긍정적인 피드백을 주세요

{context}
"""
        
        try:
            # 최근 5개 대화만 유지
            messages = self.conversation_history[-5:]
            
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system_prompt,
                messages=messages
            )
            
            assistant_response = response.content[0].text
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            
            return {
                "response": assistant_response,
                "suggestions": self._extract_suggestions(assistant_response)
            }
        except Exception as e:
            print(f"API Error: {e}")
            return {
                "response": "죄송합니다. 현재 응답할 수 없습니다.",
                "suggestions": []
            }
    
    def explain_grammar(self, sentence: str) -> str:
        """
        문법 설명 생성
        
        Args:
            sentence: 중국어 문장
            
        Returns:
            문법 설명
        """
        if self.client is None:
            return "AI 기능이 비활성화되어 있습니다."
        
        prompt = f"""
다음 중국어 문장의 문법을 초보자도 이해하기 쉽게 설명해주세요:

문장: {sentence}

다음 내용을 포함해주세요:
1. 문장 구조 분석 (주어, 동사, 목적어 등)
2. 주요 문법 포인트
3. 비슷한 예문 2개
4. 흔한 실수 주의사항

한국어로 설명해주세요.
"""
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
        except Exception as e:
            print(f"API Error: {e}")
            return "문법 설명을 생성할 수 없습니다."
    
    def generate_exercises(self, grammar_point: str, count: int = 5) -> List[Dict]:
        """
        특정 문법 포인트에 대한 연습문제 생성
        
        Args:
            grammar_point: 문법 포인트 (예: "了", "的", "是...吗?")
            count: 문제 개수
            
        Returns:
            연습문제 리스트
        """
        if self.client is None:
            return []
        
        prompt = f"""
'{grammar_point}' 문법에 대한 연습문제 {count}개를 생성해주세요.

각 문제는 다음 정보를 포함해야 합니다:
1. 문제 유형 (빈칸 채우기, 순서 배열, 번역 등)
2. 문제
3. 선택지 (객관식인 경우)
4. 정답
5. 해설

간단한 형식으로 작성해주세요.
"""
        
        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._parse_exercises(message.content[0].text)
        except Exception as e:
            print(f"API Error: {e}")
            return []
    
    def _generate_simple_dialogue(self, vocabulary: List[str]) -> List[Dict]:
        """AI 없이 간단한 대화 생성 (폴백)"""
        dialogues = []
        patterns = [
            ("你好！", "Nǐ hǎo!", "안녕하세요!"),
            ("谢谢。", "Xièxie.", "감사합니다."),
            ("再见！", "Zàijiàn!", "안녕히 가세요!"),
        ]
        
        for pattern in patterns[:min(3, len(vocabulary))]:
            dialogues.append({
                'context': '일상 인사',
                'chinese': pattern[0],
                'pinyin': pattern[1],
                'korean': pattern[2],
                'grammar_note': '기본 인사 표현'
            })
        
        return dialogues
    
    def _parse_dialogue_response(self, response: str) -> List[Dict]:
        """AI 응답에서 대화 파싱"""
        # 간단한 파싱 로직
        dialogues = []
        # 실제로는 더 정교한 파싱이 필요
        return dialogues
    
    def _extract_suggestions(self, response: str) -> List[str]:
        """응답에서 학습 제안 추출"""
        suggestions = []
        # 제안 추출 로직
        return suggestions
    
    def _parse_exercises(self, response: str) -> List[Dict]:
        """연습문제 파싱"""
        # 파싱 로직
        return []


if __name__ == "__main__":
    # 테스트
    tutor = ChineseAITutor()
    
    if tutor.client:
        print("AI Tutor initialized with API")
        
        # 대화 생성 테스트
        words = ["你好", "谢谢", "再见"]
        dialogues = tutor.generate_dialogue(words, level="beginner")
        print(f"Generated {len(dialogues)} dialogues")
        
        # 대화 연습 테스트
        response = tutor.chat_practice("你好，我是学生。")
        print(f"Response: {response['response']}")
    else:
        print("AI Tutor running in fallback mode")
