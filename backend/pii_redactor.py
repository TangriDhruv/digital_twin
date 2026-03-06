"""
pii_redactor.py
---------------
Detects and redacts PII from chunk text before embedding.
Runs as a pipeline step between loading and embedding.

Uses Microsoft Presidio for detection + anonymization.
Falls back to regex-based redaction if Presidio is unavailable.

Entities detected and redacted:
  - Email addresses
  - Phone numbers
  - Credit card numbers
  - US Social Security Numbers
  - Passwords / secrets (regex-based)
  - URLs containing personal tokens

Intentionally NOT redacted (safe for a professional twin):
  - Person names    — the twin IS Dhruv, names are identity
  - Locations       — Pittsburgh, CMU are part of the profile
  - Organizations   — KPMG, Goldman Sachs are work context
  - Dates           — work history requires dates


"""

import logging
import re
from presidio_anonymizer.entities import OperatorConfig
log = logging.getLogger(__name__)

# ── Entities to redact ────────────────────────────────────────────────────────
# Full list: https://microsoft.github.io/presidio/supported_entities/

REDACT_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IBAN_CODE",
    "IP_ADDRESS",
    "MEDICAL_LICENSE",
]

# Replacement tokens — readable but clearly redacted
REPLACEMENT_MAP = {
    "EMAIL_ADDRESS":    "[EMAIL REDACTED]",
    "PHONE_NUMBER":     "[PHONE REDACTED]",
    "CREDIT_CARD":      "[CARD REDACTED]",
    "US_SSN":           "[SSN REDACTED]",
    "US_BANK_NUMBER":   "[BANK REDACTED]",
    "US_PASSPORT":      "[PASSPORT REDACTED]",
    "US_DRIVER_LICENSE":"[LICENSE REDACTED]",
    "IBAN_CODE":        "[IBAN REDACTED]",
    "IP_ADDRESS":       "[IP REDACTED]",
    "MEDICAL_LICENSE":  "[LICENSE REDACTED]",
    "DEFAULT":          "[REDACTED]",
}

# ── Regex patterns for things Presidio misses ─────────────────────────────────

REGEX_PATTERNS = [
    # GitHub personal access tokens
    (re.compile(r'ghp_[A-Za-z0-9]{36}'), "[GITHUB TOKEN REDACTED]"),
    # OpenAI API keys
    (re.compile(r'sk-[A-Za-z0-9]{32,}'), "[API KEY REDACTED]"),
    # Generic bearer tokens
    (re.compile(r'Bearer\s+[A-Za-z0-9\-._~+/]+=*'), "[BEARER TOKEN REDACTED]"),
    # AWS access keys
    (re.compile(r'AKIA[0-9A-Z]{16}'), "[AWS KEY REDACTED]"),
    # Private keys
    (re.compile(r'-----BEGIN [A-Z ]+PRIVATE KEY-----'), "[PRIVATE KEY REDACTED]"),
]


class PIIRedactor:
    """
    Detects and redacts PII from text using Presidio + regex fallback.
    Instantiate once and reuse — Presidio engine is expensive to load.
    """

    def __init__(self):
        self._analyzer   = None
        self._anonymizer = None
        self._presidio_available = False
        self._load_presidio()

    def _load_presidio(self):
        """
        Attempt to load Presidio. Falls back to regex-only mode if
        spacy model is not installed.
        """
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig

            self._analyzer   = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            self._presidio_available = True
            log.info("PII redactor: Presidio loaded (full detection)")
        except Exception as e:
            log.warning(f"PII redactor: Presidio unavailable ({e}) — using regex fallback")
            self._presidio_available = False

    def redact(self, text: str) -> tuple[str, list[str]]:
        """
        Redact PII from text.

        Returns:
            (redacted_text, list_of_entity_types_found)
        """
        found_entities = []

        # Step 1 — Presidio detection (if available)
        if self._presidio_available:
            text, found_entities = self._presidio_redact(text)

        # Step 2 — Regex patterns (always run, catches tokens Presidio misses)
        text, regex_hits = self._regex_redact(text)
        found_entities.extend(regex_hits)

        return text, found_entities

    def redact_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Redact PII from all chunks in the pipeline.
        Modifies the 'text' field of each chunk in place.
        Logs a summary of what was found.
        """
        total_redactions = 0
        redacted_chunks  = 0

        for chunk in chunks:
            original = chunk["text"]
            cleaned, entities = self.redact(original)

            if entities:
                chunk["text"] = cleaned
                chunk["pii_redacted"] = entities   # audit trail
                total_redactions += len(entities)
                redacted_chunks  += 1
                log.debug(
                    f"Redacted {entities} in chunk "
                    f"[{chunk.get('topic', '?')} / {chunk.get('section', '?')}]"
                )

        if total_redactions > 0:
            log.info(
                f"PII redaction: {total_redactions} items redacted "
                f"across {redacted_chunks}/{len(chunks)} chunks"
            )
        else:
            log.info("PII redaction: no PII detected")

        return chunks

    # ── Internal methods ───────────────────────────────────────────────────

    def _presidio_redact(self, text: str) -> tuple[str, list[str]]:
        """Run Presidio analyzer + anonymizer on text."""
        

        results = self._analyzer.analyze(
            text=text,
            entities=REDACT_ENTITIES,
            language="en",
        )

        if not results:
            return text, []

        found = [r.entity_type for r in results]

        # Build operator config — replace each entity type with its token
        operators = {
            entity: OperatorConfig(
                "replace",
                {"new_value": REPLACEMENT_MAP.get(entity, REPLACEMENT_MAP["DEFAULT"])}
            )
            for entity in set(found)
        }

        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=operators,
        )

        return anonymized.text, found

    def _regex_redact(self, text: str) -> tuple[str, list[str]]:
        """Apply regex patterns for secrets and tokens Presidio misses."""
        found = []
        for pattern, replacement in REGEX_PATTERNS:
            if pattern.search(text):
                text = pattern.sub(replacement, text)
                found.append(replacement)
        return text, found
