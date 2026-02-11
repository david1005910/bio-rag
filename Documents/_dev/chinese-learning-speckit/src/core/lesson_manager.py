"""
레슨 관리 모듈
단어 학습, 퀴즈 생성, 대화 관리
"""

import random
from typing import List, Dict, Optional


class LessonManager:
    """레슨 및 학습 콘텐츠 관리"""
    
    def __init__(self, vocabulary_data: List[Dict]):
        """
        Args:
            vocabulary_data: 단어 데이터 리스트
        """
        self.vocabulary = vocabulary_data
        self.current_lesson = 0
        self.learned_words = set()
    
    def get_lesson(self, lesson_num: int, words_per_lesson: int = 10) -> List[Dict]:
        """
        레슨별 단어 가져오기
        
        Args:
            lesson_num: 레슨 번호 (0부터 시작)
            words_per_lesson: 레슨당 단어 수
            
        Returns:
            단어 리스트
        """
        start = lesson_num * words_per_lesson
        end = start + words_per_lesson
        
        if start >= len(self.vocabulary):
            return []
        
        return self.vocabulary[start:end]
    
    def create_dialogue(self, words: List[Dict]) -> List[Dict]:
        """
        학습 단어로 대화문 생성
        
        Args:
            words: 단어 리스트
            
        Returns:
            대화문 리스트
        """
        dialogues = []
        
        # 간단한 패턴으로 대화 생성
        patterns = [
            ("你好！{word}怎么样？", "Nǐ hǎo! {pinyin} zěnmeyàng?", "안녕하세요! {meaning} 어떠세요?"),
            ("我喜欢{word}。", "Wǒ xǐhuan {pinyin}.", "저는 {meaning}를 좋아합니다."),
            ("{word}在哪里？", "{pinyin} zài nǎlǐ?", "{meaning}가 어디에 있나요?"),
            ("这是{word}吗？", "Zhè shì {pinyin} ma?", "이것이 {meaning}인가요?"),
            ("我想要{word}。", "Wǒ xiǎng yào {pinyin}.", "저는 {meaning}를 원합니다."),
        ]
        
        for i, word in enumerate(words[:5]):  # 처음 5개 단어만 사용
            pattern = patterns[i % len(patterns)]
            
            dialogues.append({
                'chinese': pattern[0].format(word=word['simplified']),
                'pinyin': pattern[1].format(pinyin=word['pinyin']),
                'korean': pattern[2].format(meaning=word['definitions'][0])
            })
        
        return dialogues
    
    def generate_quiz(self, 
                     words: List[Dict], 
                     quiz_type: str = 'translation',
                     num_questions: int = 5) -> List[Dict]:
        """
        퀴즈 생성
        
        Args:
            words: 단어 리스트
            quiz_type: 퀴즈 유형 ('translation', 'listening', 'fill_blank')
            num_questions: 문제 개수
            
        Returns:
            퀴즈 문제 리스트
        """
        quiz = []
        selected_words = random.sample(words, min(num_questions, len(words)))
        
        for word in selected_words:
            if quiz_type == 'translation':
                question = self._create_translation_question(word)
            elif quiz_type == 'listening':
                question = self._create_listening_question(word)
            elif quiz_type == 'fill_blank':
                question = self._create_fill_blank_question(word)
            else:
                question = self._create_translation_question(word)
            
            quiz.append(question)
        
        return quiz
    
    def _create_translation_question(self, word: Dict) -> Dict:
        """번역 문제 생성"""
        options = [word['definitions'][0]]
        
        # 다른 단어들에서 오답 선택지 생성
        other_words = [w for w in self.vocabulary if w != word]
        wrong_answers = random.sample(other_words, min(3, len(other_words)))
        options.extend([w['definitions'][0] for w in wrong_answers])
        
        random.shuffle(options)
        
        return {
            'type': 'translation',
            'question': f"'{word['simplified']}'의 의미는?",
            'word': word['simplified'],
            'pinyin': word['pinyin'],
            'answer': word['definitions'][0],
            'options': options
        }
    
    def _create_listening_question(self, word: Dict) -> Dict:
        """듣기 문제 생성"""
        options = [word['simplified']]
        
        other_words = [w for w in self.vocabulary if w != word]
        wrong_answers = random.sample(other_words, min(3, len(other_words)))
        options.extend([w['simplified'] for w in wrong_answers])
        
        random.shuffle(options)
        
        return {
            'type': 'listening',
            'question': '들려주는 단어는?',
            'audio_word': word['simplified'],
            'pinyin': word['pinyin'],
            'answer': word['simplified'],
            'options': options
        }
    
    def _create_fill_blank_question(self, word: Dict) -> Dict:
        """빈칸 채우기 문제 생성"""
        # 간단한 문장 생성
        sentence_patterns = [
            f"我是__。 (Wǒ shì __.) → 나는 {word['definitions'][0]}입니다.",
            f"这是__。 (Zhè shì __.) → 이것은 {word['definitions'][0]}입니다.",
            f"我喜欢__。 (Wǒ xǐhuan __.) → 나는 {word['definitions'][0]}를 좋아합니다.",
        ]
        
        sentence = random.choice(sentence_patterns)
        
        options = [word['simplified']]
        other_words = [w for w in self.vocabulary if w != word]
        wrong_answers = random.sample(other_words, min(3, len(other_words)))
        options.extend([w['simplified'] for w in wrong_answers])
        
        random.shuffle(options)
        
        return {
            'type': 'fill_blank',
            'question': f"빈칸에 들어갈 단어는?\n{sentence}",
            'answer': word['simplified'],
            'options': options
        }
    
    def mark_as_learned(self, word_id: str) -> None:
        """단어를 학습 완료로 표시"""
        self.learned_words.add(word_id)
    
    def get_learned_count(self) -> int:
        """학습 완료한 단어 수"""
        return len(self.learned_words)
    
    def get_progress(self) -> float:
        """
        전체 학습 진도율
        
        Returns:
            진도율 (0.0 ~ 1.0)
        """
        if len(self.vocabulary) == 0:
            return 0.0
        return len(self.learned_words) / len(self.vocabulary)


if __name__ == "__main__":
    # 테스트
    from data_parser import ChineseDataParser
    
    parser = ChineseDataParser()
    words = parser.load_hsk_words(level=1)
    
    manager = LessonManager(words)
    
    # 레슨 가져오기
    lesson = manager.get_lesson(0, words_per_lesson=5)
    print(f"Lesson 1: {len(lesson)} words")
    for word in lesson:
        print(f"  {word['simplified']} - {word['definitions'][0]}")
    
    # 대화 생성
    print("\n대화 예시:")
    dialogues = manager.create_dialogue(lesson)
    for dialogue in dialogues:
        print(f"  中: {dialogue['chinese']}")
        print(f"  拼: {dialogue['pinyin']}")
        print(f"  韓: {dialogue['korean']}\n")
    
    # 퀴즈 생성
    print("퀴즈:")
    quiz = manager.generate_quiz(lesson, num_questions=3)
    for i, q in enumerate(quiz, 1):
        print(f"\nQ{i}. {q['question']}")
        for j, option in enumerate(q['options'], 1):
            print(f"  {j}) {option}")
        print(f"  정답: {q['answer']}")
