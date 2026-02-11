"""
중국어 학습 프로그램 - CLI 진입점
"""

import sys
import os

# src 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(__file__))

from src.core.data_parser import ChineseDataParser
from src.core.lesson_manager import LessonManager
from src.core.progress_tracker import ProgressTracker
from src.speech.speech_handler import SpeechHandler


class ChineseLearningApp:
    """중국어 학습 앱 CLI"""
    
    def __init__(self):
        self.parser = ChineseDataParser()
        self.speech = SpeechHandler()
        self.progress = ProgressTracker()
        self.lesson_manager = None
    
    def setup(self):
        """초기 설정"""
        print("=== 중국어 학습 프로그램 ===\n")
        print("데이터 로딩 중...")
        
        # HSK 단어 로드
        vocabulary = self.parser.load_hsk_words(level=1)
        self.lesson_manager = LessonManager(vocabulary)
        
        print(f"{len(vocabulary)}개 단어 로드 완료!\n")
    
    def start_lesson(self, lesson_num: int = 0):
        """레슨 시작"""
        words = self.lesson_manager.get_lesson(lesson_num, words_per_lesson=10)
        
        if not words:
            print("이 레슨에는 단어가 없습니다.")
            return
        
        print(f"\n=== Lesson {lesson_num + 1} ===")
        print(f"{len(words)}개의 단어를 학습합니다.\n")
        
        # 세션 시작
        session_id = self.progress.start_session(lesson_num)
        
        # 단어 학습
        for i, word in enumerate(words, 1):
            print(f"{i}. {word['simplified']} ({word['pinyin']})")
            print(f"   의미: {', '.join(word['definitions'][:3])}\n")
            
            # 음성 재생
            audio_file = self.speech.text_to_speech(word['simplified'])
            if audio_file:
                self.speech.play_audio(audio_file)
            
            input("다음 단어로 넘어가려면 Enter를 누르세요...")
            print()
        
        # 대화 연습
        print("\n=== 회화 연습 ===\n")
        dialogues = self.lesson_manager.create_dialogue(words)
        
        for dialogue in dialogues:
            print(f"중국어: {dialogue['chinese']}")
            print(f"병음: {dialogue['pinyin']}")
            print(f"한국어: {dialogue['korean']}\n")
            
            audio_file = self.speech.text_to_speech(dialogue['chinese'])
            if audio_file:
                self.speech.play_audio(audio_file)
            
            input()
        
        # 퀴즈
        self.start_quiz(words, session_id)
    
    def start_quiz(self, words, session_id):
        """퀴즈 시작"""
        print("\n=== 퀴즈 시간 ===\n")
        
        quiz = self.lesson_manager.generate_quiz(words, num_questions=5)
        score = 0
        
        for i, q in enumerate(quiz, 1):
            print(f"Q{i}. {q['question']}")
            
            for j, option in enumerate(q['options'], 1):
                print(f"  {j}) {option}")
            
            try:
                answer = input("\n답 번호: ")
                selected = q['options'][int(answer) - 1]
                
                if selected == q['answer']:
                    print("정답! ✓\n")
                    score += 1
                else:
                    print(f"오답. 정답: {q['answer']}\n")
            except (ValueError, IndexError):
                print("잘못된 입력\n")
        
        percentage = (score / len(quiz)) * 100
        print(f"점수: {score}/{len(quiz)} ({percentage:.1f}%)")
        
        # 세션 종료
        self.progress.end_session(session_id, len(words), percentage)
    
    def show_statistics(self):
        """통계 표시"""
        print("\n=== 학습 통계 ===\n")
        
        stats = self.progress.get_statistics()
        
        print(f"총 학습 시간: {stats['total_study_hours']:.1f} 시간")
        print(f"마스터한 단어: {stats['mastered_words']} 개")
        print(f"평균 퀴즈 점수: {stats['average_quiz_score']:.1f}%")
        print(f"평균 발음 점수: {stats['average_pronunciation']:.1f}%")
    
    def run(self):
        """메인 루프"""
        self.setup()
        
        while True:
            print("\n=== 메뉴 ===")
            print("1. 새 레슨 시작")
            print("2. 통계 확인")
            print("3. 종료")
            
            choice = input("\n선택: ")
            
            if choice == '1':
                try:
                    lesson_num = int(input("레슨 번호 (0부터 시작): "))
                    self.start_lesson(lesson_num)
                except ValueError:
                    print("잘못된 입력입니다.")
            
            elif choice == '2':
                self.show_statistics()
            
            elif choice == '3':
                print("\n학습을 종료합니다. 수고하셨습니다! 👋")
                self.progress.close()
                break
            
            else:
                print("잘못된 선택입니다.")


def main():
    """메인 함수"""
    try:
        app = ChineseLearningApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
