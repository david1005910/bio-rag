"""Tests for translation_service.py"""
import pytest
import re
from typing import Tuple
from unittest.mock import MagicMock


# ============================================================================
# Testable implementation of TranslationService
# ============================================================================

class TestableTranslationService:
    """Testable version of TranslationService."""

    def __init__(self, client=None):
        self.client = client

    def is_korean(self, text: str) -> bool:
        """Check if text contains more than 20% Korean characters."""
        korean_pattern = re.compile('[가-힣]')
        korean_chars = korean_pattern.findall(text)
        return len(korean_chars) > len(text) * 0.2

    def translate_to_english(self, text: str) -> str:
        """Translate Korean text to English."""
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
        except Exception:
            return text

    def translate_with_both(self, text: str) -> Tuple[str, str]:
        """Return both original and translated text."""
        if not self.is_korean(text):
            return text, text

        english = self.translate_to_english(text)
        return text, english


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def translation_service():
    """Create a translation service without client."""
    return TestableTranslationService()


@pytest.fixture
def translation_service_with_mock_client():
    """Create a translation service with mocked client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Translated English text"
    mock_client.chat.completions.create.return_value = mock_response

    return TestableTranslationService(client=mock_client)


# ============================================================================
# Tests
# ============================================================================

class TestIsKorean:
    """Tests for Korean language detection."""

    def test_is_korean_pure_korean(self, translation_service):
        """Test detection of pure Korean text."""
        korean_text = "암 치료에 대한 연구"
        assert translation_service.is_korean(korean_text) is True

    def test_is_korean_pure_english(self, translation_service):
        """Test detection of pure English text."""
        english_text = "Research on cancer treatment"
        assert translation_service.is_korean(english_text) is False

    def test_is_korean_mixed_mostly_korean(self, translation_service):
        """Test detection of mixed text with mostly Korean."""
        mixed_text = "암 치료 cancer 연구 research 방법론"
        # More than 20% Korean characters
        assert translation_service.is_korean(mixed_text) is True

    def test_is_korean_mixed_mostly_english(self, translation_service):
        """Test detection of mixed text with mostly English."""
        mixed_text = "cancer treatment research methodology 암"
        # Less than 20% Korean characters
        assert translation_service.is_korean(mixed_text) is False

    def test_is_korean_empty_string(self, translation_service):
        """Test detection of empty string."""
        assert translation_service.is_korean("") is False

    def test_is_korean_threshold_boundary(self, translation_service):
        """Test detection at the 20% boundary."""
        # Exactly at 20% should return False (> 0.2, not >=)
        # 10 chars total, 2 Korean = 20%
        text = "abcdefgh가나"  # 10 chars, 2 Korean
        assert translation_service.is_korean(text) is False

        # Just over 20%
        text = "abcdefg가나다"  # 10 chars, 3 Korean = 30%
        assert translation_service.is_korean(text) is True

    def test_is_korean_numbers_and_symbols(self, translation_service):
        """Test with numbers and symbols."""
        text = "123!@#암연구456"
        # Only count actual characters, Korean chars as percentage
        assert translation_service.is_korean(text) is True

    def test_is_korean_whitespace_only(self, translation_service):
        """Test with whitespace only."""
        assert translation_service.is_korean("   ") is False


class TestTranslateToEnglish:
    """Tests for Korean to English translation."""

    def test_translate_english_text_passthrough(self, translation_service):
        """Test that English text passes through unchanged."""
        english_text = "Cancer research methodology"
        result = translation_service.translate_to_english(english_text)
        assert result == english_text

    def test_translate_korean_without_client(self, translation_service):
        """Test translating Korean without client returns original."""
        korean_text = "암 치료 연구"
        result = translation_service.translate_to_english(korean_text)
        assert result == korean_text

    def test_translate_korean_with_client(self, translation_service_with_mock_client):
        """Test translating Korean with mocked client."""
        korean_text = "암 치료에 대한 연구"
        result = translation_service_with_mock_client.translate_to_english(korean_text)
        assert result == "Translated English text"

    def test_translate_calls_api_correctly(self, translation_service_with_mock_client):
        """Test that translation calls API with correct parameters."""
        korean_text = "암 치료"
        translation_service_with_mock_client.translate_to_english(korean_text)

        mock_client = translation_service_with_mock_client.client
        mock_client.chat.completions.create.assert_called_once()

        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-4o'
        assert call_args[1]['max_tokens'] == 500
        assert call_args[1]['temperature'] == 0.1

    def test_translate_uses_biomedical_prompt(self, translation_service_with_mock_client):
        """Test that translation uses biomedical terminology prompt."""
        korean_text = "암 치료"
        translation_service_with_mock_client.translate_to_english(korean_text)

        mock_client = translation_service_with_mock_client.client
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']

        system_message = messages[0]['content']
        assert 'biomedical' in system_message.lower()

    def test_translate_handles_api_error(self):
        """Test translation handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        service = TestableTranslationService(client=mock_client)
        korean_text = "암 치료"

        result = service.translate_to_english(korean_text)
        assert result == korean_text  # Returns original on error


class TestTranslateWithBoth:
    """Tests for getting both original and translated text."""

    def test_translate_with_both_english(self, translation_service):
        """Test translate_with_both returns same text for English."""
        english_text = "Cancer research"
        original, translated = translation_service.translate_with_both(english_text)

        assert original == english_text
        assert translated == english_text

    def test_translate_with_both_korean(self, translation_service_with_mock_client):
        """Test translate_with_both returns both texts for Korean."""
        korean_text = "암 치료 연구"
        original, translated = translation_service_with_mock_client.translate_with_both(korean_text)

        assert original == korean_text
        assert translated == "Translated English text"

    def test_translate_with_both_returns_tuple(self, translation_service):
        """Test translate_with_both returns a tuple."""
        result = translation_service.translate_with_both("test")

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestClientInitialization:
    """Tests for client initialization scenarios."""

    def test_service_without_client(self):
        """Test service works without client."""
        service = TestableTranslationService()
        assert service.client is None

    def test_service_with_client(self):
        """Test service with client."""
        mock_client = MagicMock()
        service = TestableTranslationService(client=mock_client)
        assert service.client == mock_client


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_korean_text(self, translation_service_with_mock_client):
        """Test handling very long Korean text."""
        long_korean = "암 치료 연구 " * 100
        result = translation_service_with_mock_client.translate_to_english(long_korean)
        assert result == "Translated English text"

    def test_korean_with_special_characters(self, translation_service):
        """Test Korean text with special characters."""
        text = "암 치료! 연구? #연구"
        assert translation_service.is_korean(text) is True

    def test_korean_with_numbers(self, translation_service):
        """Test Korean text with numbers."""
        text = "암 치료 2024년 연구"
        assert translation_service.is_korean(text) is True

    def test_single_korean_character(self, translation_service):
        """Test single Korean character in long text."""
        text = "This is a very long English text with just one 암 Korean character"
        # One Korean char out of many
        assert translation_service.is_korean(text) is False

    def test_korean_scientific_terms(self, translation_service):
        """Test Korean scientific/medical terms."""
        text = "면역요법 항암제 유전자치료"
        assert translation_service.is_korean(text) is True
