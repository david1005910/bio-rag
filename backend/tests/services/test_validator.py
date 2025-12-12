"""Tests for response validator"""

import pytest

from app.services.rag.retriever import RetrievedDocument
from app.services.rag.validator import ResponseValidator, ValidationResult


@pytest.fixture
def validator():
    """Create validator instance"""
    return ResponseValidator(similarity_threshold=0.5, min_citations=1)


@pytest.fixture
def sample_documents():
    """Create sample retrieved documents"""
    return [
        RetrievedDocument(
            chunk_id="chunk1",
            pmid="12345678",
            title="Test Paper 1",
            content="This is the content of test paper 1.",
            section="abstract",
            score=0.9,
            metadata={},
        ),
        RetrievedDocument(
            chunk_id="chunk2",
            pmid="87654321",
            title="Test Paper 2",
            content="This is the content of test paper 2.",
            section="methods",
            score=0.8,
            metadata={},
        ),
    ]


class TestPMIDExtraction:
    """Tests for PMID extraction"""

    def test_extract_pmid_standard_format(self, validator):
        """Test extracting PMID in standard format"""
        text = "According to the study [PMID:12345678], the treatment is effective."
        pmids = validator._extract_pmids(text)
        assert pmids == ["12345678"]

    def test_extract_pmid_with_space(self, validator):
        """Test extracting PMID with space"""
        text = "The research [PMID: 12345678] shows promising results."
        pmids = validator._extract_pmids(text)
        assert pmids == ["12345678"]

    def test_extract_multiple_pmids(self, validator):
        """Test extracting multiple PMIDs"""
        text = "Studies [PMID:12345678] and [PMID:87654321] both confirm this."
        pmids = validator._extract_pmids(text)
        assert "12345678" in pmids
        assert "87654321" in pmids

    def test_extract_duplicate_pmids(self, validator):
        """Test that duplicate PMIDs are removed"""
        text = "Study [PMID:12345678] confirms [PMID:12345678] this finding."
        pmids = validator._extract_pmids(text)
        assert pmids == ["12345678"]  # Only one

    def test_no_pmids(self, validator):
        """Test text with no PMIDs"""
        text = "This is a response without any citations."
        pmids = validator._extract_pmids(text)
        assert pmids == []


class TestValidation:
    """Tests for response validation"""

    def test_valid_response(self, validator, sample_documents):
        """Test validation of valid response"""
        response = "The treatment is effective [PMID:12345678]."
        result = validator.validate(response, sample_documents)

        assert result.is_valid
        assert "12345678" in result.valid_citations
        assert len(result.invalid_citations) == 0

    def test_invalid_citation(self, validator, sample_documents):
        """Test validation with invalid citation"""
        response = "According to [PMID:99999999], this is true."
        result = validator.validate(response, sample_documents)

        assert not result.is_valid
        assert "99999999" in result.invalid_citations

    def test_mixed_citations(self, validator, sample_documents):
        """Test validation with mixed valid and invalid citations"""
        response = "Study [PMID:12345678] shows X, while [PMID:99999999] shows Y."
        result = validator.validate(response, sample_documents)

        assert not result.is_valid
        assert "12345678" in result.valid_citations
        assert "99999999" in result.invalid_citations

    def test_no_citations_warning(self, validator, sample_documents):
        """Test warning when no citations"""
        validator.min_citations = 1
        response = "The treatment is effective without any citations."
        result = validator.validate(response, sample_documents)

        assert not result.is_valid
        assert len(result.warnings) > 0


class TestHallucinationDetection:
    """Tests for hallucination detection"""

    def test_claim_with_citation(self, validator, sample_documents):
        """Test claim with citation is not flagged"""
        response = "Research indicates that this treatment works [PMID:12345678]."
        has_hallucination, suspicious = validator.check_hallucination(
            response, sample_documents
        )
        assert not has_hallucination

    def test_claim_without_citation(self, validator, sample_documents):
        """Test claim without citation is flagged"""
        response = "Research indicates that this treatment is very effective in all patients."
        has_hallucination, suspicious = validator.check_hallucination(
            response, sample_documents
        )
        assert has_hallucination
        assert len(suspicious) > 0

    def test_multiple_uncited_claims(self, validator, sample_documents):
        """Test multiple uncited claims"""
        response = """
        Study shows that treatment A is effective.
        Research indicates that treatment B has side effects.
        Evidence suggests that combination therapy works best.
        """
        has_hallucination, suspicious = validator.check_hallucination(
            response, sample_documents
        )
        assert has_hallucination
        assert len(suspicious) >= 2


class TestResponseCleaning:
    """Tests for response cleaning"""

    def test_clean_invalid_citations(self, validator, sample_documents):
        """Test cleaning invalid citations"""
        response = "According to [PMID:99999999], this is the result."
        result = ValidationResult(
            is_valid=False,
            confidence_score=0.5,
            cited_pmids=["99999999"],
            valid_citations=[],
            invalid_citations=["99999999"],
            warnings=["Invalid citation"],
        )

        cleaned = validator.clean_response(response, result)
        assert "[citation needed]" in cleaned
        assert "99999999" not in cleaned

    def test_add_disclaimer(self, validator, sample_documents):
        """Test adding disclaimer for warnings"""
        response = "This is the response."
        result = ValidationResult(
            is_valid=True,
            confidence_score=0.5,
            cited_pmids=[],
            valid_citations=[],
            invalid_citations=[],
            warnings=["Some warning"],
        )

        cleaned = validator.clean_response(response, result)
        assert "Note:" in cleaned or "⚠️" in cleaned
