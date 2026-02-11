"""
데이터 파싱 모듈
CC-CEDICT 사전 파싱 및 병음 변환
"""

import re
from typing import List, Dict, Optional
from pypinyin import pinyin, Style


class ChineseDataParser:
    """중국어 데이터 파서"""
    
    def __init__(self):
        self.vocabulary: List[Dict] = []
    
    def parse_cedict(self, file_path: str, limit: int = 500) -> List[Dict]:
        """
        CC-CEDICT 파일 파싱
        
        Args:
            file_path: CEDICT 파일 경로
            limit: 최대 단어 수 (HSK 1-2급 수준)
            
        Returns:
            단어 리스트
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                # 정규식으로 파싱
                # 형식: 传统字 简体字 [pin1 yin1] /definition1/definition2/
                match = re.match(
                    r'(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/(.+)/', 
                    line
                )
                
                if match:
                    traditional, simplified, pinyin_str, definitions = match.groups()
                    
                    self.vocabulary.append({
                        'simplified': simplified,
                        'traditional': traditional,
                        'pinyin': pinyin_str,
                        'definitions': definitions.split('/'),
                        'level': self._estimate_level(simplified)
                    })
                    
                    if len(self.vocabulary) >= limit:
                        break
        
        return self.vocabulary
    
    def get_pinyin(self, chinese_text: str, with_tone: bool = True) -> str:
        """
        한자를 병음으로 변환
        
        Args:
            chinese_text: 중국어 텍스트
            with_tone: 성조 표시 여부
            
        Returns:
            병음 문자열
        """
        if with_tone:
            style = Style.TONE
        else:
            style = Style.NORMAL
        
        pinyin_list = pinyin(chinese_text, style=style)
        return ' '.join([''.join(p) for p in pinyin_list])
    
    def _estimate_level(self, word: str) -> str:
        """
        단어 난이도 추정
        
        Args:
            word: 중국어 단어
            
        Returns:
            'HSK1', 'HSK2', 'HSK3' 등
        """
        # 단순한 추정: 글자 수로 판단
        if len(word) <= 2:
            return 'HSK1'
        elif len(word) <= 3:
            return 'HSK2'
        else:
            return 'HSK3'
    
    def load_hsk_words(self, level: int = 1) -> List[Dict]:
        """
        HSK 레벨별 단어 로드 (data/vocabulary.json 에서)

        Args:
            level: HSK 레벨 (1-6)

        Returns:
            단어 리스트
        """
        import json
        import os

        # data/vocabulary.json 경로 탐색 (실행 위치 무관)
        candidates = [
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vocabulary.json'),
            'data/vocabulary.json',
        ]

        for path in candidates:
            path = os.path.normpath(path)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        all_words = json.load(f)
                    target = f'HSK{level}'
                    filtered = [w for w in all_words if w.get('level', '').upper() == target]
                    return filtered if filtered else all_words
                except Exception:
                    pass

        # 파일이 없으면 기본 5개 반환
        return [
            {'simplified': '你好', 'traditional': '你好', 'pinyin': 'nǐ hǎo',
             'definitions': ['안녕하세요', 'hello'], 'level': 'HSK1'},
            {'simplified': '谢谢', 'traditional': '謝謝', 'pinyin': 'xiè xie',
             'definitions': ['감사합니다', 'thank you'], 'level': 'HSK1'},
            {'simplified': '再见', 'traditional': '再見', 'pinyin': 'zài jiàn',
             'definitions': ['안녕히 가세요', 'goodbye'], 'level': 'HSK1'},
        ]
    
    def search_word(self, query: str) -> Optional[Dict]:
        """
        단어 검색
        
        Args:
            query: 검색어 (한자 또는 병음)
            
        Returns:
            단어 정보 또는 None
        """
        for word in self.vocabulary:
            if word['simplified'] == query or word['pinyin'] == query:
                return word
        return None


if __name__ == "__main__":
    # 테스트
    parser = ChineseDataParser()
    
    # HSK 단어 로드
    words = parser.load_hsk_words(level=1)
    print(f"Loaded {len(words)} HSK1 words")
    
    for word in words:
        print(f"{word['simplified']} ({word['pinyin']}) - {word['definitions'][0]}")
    
    # 병음 변환 테스트
    text = "你好，我是学生"
    pinyin_text = parser.get_pinyin(text)
    print(f"\n{text} -> {pinyin_text}")
