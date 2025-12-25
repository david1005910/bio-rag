import os
import re
from typing import Optional, Tuple
from openai import OpenAI

class TranslationService:
    def __init__(self):
        self._init_client()
    
    def _init_client(self):
        base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
        api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
        
        if base_url and api_key:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                self.client = OpenAI(api_key=openai_key)
            else:
                self.client = None
    
    def is_korean(self, text: str) -> bool:
        korean_pattern = re.compile('[가-힣]')
        korean_chars = korean_pattern.findall(text)
        return len(korean_chars) > len(text) * 0.2
    
    def translate_to_english(self, text: str) -> str:
        if not self.is_korean(text):
            return text
        
        if not self.client:
            return text
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a biomedical translator. Translate the Korean text to English for PubMed search. Use proper biomedical/scientific terminology. Return ONLY the English translation, nothing else."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def translate_with_both(self, text: str) -> Tuple[str, str]:
        if not self.is_korean(text):
            return text, text
        
        english = self.translate_to_english(text)
        return text, english
