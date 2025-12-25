import os
import re
from typing import Optional, Tuple

class TranslationService:
    def __init__(self):
        self._openai_client = None
        self._google_translator = None
        self._init_clients()

    def _init_clients(self):
        """Initialize translation clients (OpenAI and/or Google Translate)."""
        # Try OpenAI first
        try:
            from openai import OpenAI
            base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
            api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")

            if base_url and api_key:
                self._openai_client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                openai_key = os.environ.get("OPENAI_API_KEY")
                if openai_key:
                    self._openai_client = OpenAI(api_key=openai_key)
        except ImportError:
            pass

        # Initialize Google Translate as fallback
        try:
            from deep_translator import GoogleTranslator
            self._google_translator = GoogleTranslator(source='ko', target='en')
        except ImportError:
            pass

    @property
    def client(self):
        """Backward compatibility for client property."""
        return self._openai_client
    
    def is_korean(self, text: str) -> bool:
        korean_pattern = re.compile('[가-힣]')
        korean_chars = korean_pattern.findall(text)
        return len(korean_chars) > len(text) * 0.2
    
    def translate_to_english(self, text: str) -> str:
        """Translate Korean text to English for PubMed search."""
        if not self.is_korean(text):
            return text

        # Try OpenAI first (better for biomedical terms)
        if self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
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
                translated = response.choices[0].message.content.strip()
                print(f"OpenAI translation: {text} -> {translated}")
                return translated
            except Exception as e:
                print(f"OpenAI translation error: {e}")

        # Fallback to Google Translate
        if self._google_translator:
            try:
                translated = self._google_translator.translate(text)
                print(f"Google translation: {text} -> {translated}")
                return translated
            except Exception as e:
                print(f"Google translation error: {e}")

        # Return original text if no translation available
        print(f"No translation available for: {text}")
        return text
    
    def translate_with_both(self, text: str) -> Tuple[str, str]:
        if not self.is_korean(text):
            return text, text
        
        english = self.translate_to_english(text)
        return text, english
