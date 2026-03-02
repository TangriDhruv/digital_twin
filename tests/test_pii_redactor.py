"""
test_pii_redactor.py
--------------------
Unit tests for PIIRedactor — both regex layer and Presidio layer.
Tests verify that sensitive data is redacted and safe data is preserved.
"""

import pytest
from pii_redactor import PIIRedactor


@pytest.fixture(scope="module")
def redactor():
    """Single PIIRedactor instance shared across tests — Presidio is slow to load."""
    return PIIRedactor()


# ── Regex layer (always active) ───────────────────────────────────────────────

class TestRegexRedaction:

    def test_github_token_is_redacted(self, redactor):
        text = "My token is ghp_abcdefghijklmnopqrstuvwxyz1234567890AB"
        result, entities = redactor.redact(text)
        assert "ghp_" not in result
        assert "[GITHUB TOKEN REDACTED]" in result

    def test_openai_api_key_is_redacted(self, redactor):
        text = "Key: sk-abcdefghijklmnopqrstuvwxyz123456789012"
        result, entities = redactor.redact(text)
        assert "sk-" not in result
        assert "[API KEY REDACTED]" in result

    def test_aws_key_is_redacted(self, redactor):
        text = "AWS key: AKIAIOSFODNN7EXAMPLE"
        result, entities = redactor.redact(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[AWS KEY REDACTED]" in result

    def test_bearer_token_is_redacted(self, redactor):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result, entities = redactor.redact(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "[BEARER TOKEN REDACTED]" in result

    def test_private_key_header_is_redacted(self, redactor):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA..."
        result, entities = redactor.redact(text)
        assert "-----BEGIN RSA PRIVATE KEY-----" not in result
        assert "[PRIVATE KEY REDACTED]" in result

    def test_clean_text_is_unchanged(self, redactor):
        text = "I worked at KPMG as a data scientist for 4 years."
        result, entities = redactor.redact(text)
        assert result == text
        assert entities == []


# ── Safe data — must NOT be redacted ─────────────────────────────────────────

class TestSafeDataPreserved:

    def test_person_name_not_redacted(self, redactor):
        text = "My name is Dhruv Tangri."
        result, _ = redactor.redact(text)
        assert "Dhruv" in result

    def test_organisation_not_redacted(self, redactor):
        text = "I worked at Goldman Sachs and KPMG."
        result, _ = redactor.redact(text)
        assert "Goldman Sachs" in result
        assert "KPMG" in result

    def test_location_not_redacted(self, redactor):
        text = "I live in Pittsburgh and studied at CMU."
        result, _ = redactor.redact(text)
        assert "Pittsburgh" in result
        assert "CMU" in result

    def test_date_not_redacted(self, redactor):
        text = "I graduated in May 2023."
        result, _ = redactor.redact(text)
        assert "2023" in result


# ── redact() return value ─────────────────────────────────────────────────────

class TestRedactReturnValue:

    def test_returns_tuple_of_text_and_list(self, redactor):
        result = redactor.redact("Some clean text.")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], list)

    def test_entities_list_populated_on_hit(self, redactor):
        text = "Token: ghp_abcdefghijklmnopqrstuvwxyz1234567890AB"
        _, entities = redactor.redact(text)
        assert len(entities) > 0


# ── redact_chunks() ───────────────────────────────────────────────────────────

class TestRedactChunks:

    def test_redacts_text_field_in_chunks(self, redactor):
        # Use a GitHub token — always caught by the regex layer regardless of Presidio
        chunks = [
            {"text": "Token: ghp_abcdefghijklmnopqrstuvwxyz1234567890AB", "topic": "contact"},
            {"text": "I worked at KPMG.", "topic": "career"},
        ]
        result = redactor.redact_chunks(chunks)
        assert "ghp_" not in result[0]["text"]
        assert "[GITHUB TOKEN REDACTED]" in result[0]["text"]

    def test_adds_pii_redacted_audit_field(self, redactor):
        chunks = [{"text": "Token: ghp_abcdefghijklmnopqrstuvwxyz1234567890AB", "topic": "test"}]
        result = redactor.redact_chunks(chunks)
        assert "pii_redacted" in result[0]
        assert len(result[0]["pii_redacted"]) > 0

    def test_clean_chunks_have_no_audit_field(self, redactor):
        chunks = [{"text": "I studied at Carnegie Mellon University.", "topic": "education"}]
        result = redactor.redact_chunks(chunks)
        assert "pii_redacted" not in result[0]

    def test_returns_all_chunks_regardless(self, redactor):
        chunks = [
            {"text": "Clean text.", "topic": "a"},
            {"text": "Another clean sentence.", "topic": "b"},
        ]
        result = redactor.redact_chunks(chunks)
        assert len(result) == 2
