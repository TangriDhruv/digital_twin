"""
test_chunkers.py
----------------
Unit tests for ParagraphChunker and SectionChunker.
"""

import pytest
from chunkers import ParagraphChunker, SectionChunker


# ── ParagraphChunker ──────────────────────────────────────────────────────────

class TestParagraphChunker:

    def test_splits_on_blank_lines(self):
        chunker = ParagraphChunker(chunk_size=300, overlap=0)
        text = "First paragraph here.\n\nSecond paragraph here."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1  # Both fit in one chunk under 300 words
        assert "First paragraph" in chunks[0]
        assert "Second paragraph" in chunks[0]

    def test_splits_when_chunk_size_exceeded(self):
        chunker = ParagraphChunker(chunk_size=5, overlap=0)
        # Each paragraph has 5 words — adding a second would exceed chunk_size=5
        text = "one two three four five\n\nsix seven eight nine ten"
        chunks = chunker.chunk(text)
        assert len(chunks) == 2
        assert "one" in chunks[0]
        assert "six" in chunks[1]

    def test_overlap_carries_words_forward(self):
        chunker = ParagraphChunker(chunk_size=5, overlap=2)
        text = "alpha bravo charlie delta epsilon\n\nfoxtrot golf hotel india juliet"
        chunks = chunker.chunk(text)
        # Second chunk should contain the last 2 words of the first chunk
        assert "delta" in chunks[1] or "epsilon" in chunks[1]

    def test_empty_text_returns_empty_list(self):
        chunker = ParagraphChunker()
        assert chunker.chunk("") == []

    def test_single_paragraph_returns_one_chunk(self):
        chunker = ParagraphChunker()
        text = "Just one paragraph with no blank lines."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == "Just one paragraph with no blank lines."

    def test_strips_whitespace_from_paragraphs(self):
        chunker = ParagraphChunker()
        text = "   leading spaces   \n\n   another paragraph   "
        chunks = chunker.chunk(text)
        assert all(not c.startswith(" ") for c in chunks)

    def test_ignores_blank_only_paragraphs(self):
        chunker = ParagraphChunker()
        text = "First paragraph.\n\n   \n\nSecond paragraph."
        chunks = chunker.chunk(text)
        # Blank-only paragraph should be ignored, not create an empty chunk
        assert all(c.strip() for c in chunks)


# ── SectionChunker ────────────────────────────────────────────────────────────

class TestSectionChunker:

    def test_splits_on_double_hash_headings(self):
        chunker = SectionChunker()
        text = "## Skills\nPython, SQL\n## Education\nCMU, MS"
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_prefixes_chunks_with_section_name(self):
        chunker = SectionChunker()
        text = "## Skills\nPython, SQL"
        chunks = chunker.chunk(text)
        assert chunks[0].startswith("[Section: Skills]")

    def test_skips_empty_sections(self):
        chunker = SectionChunker()
        text = "## EmptySection\n\n## Skills\nPython"
        chunks = chunker.chunk(text)
        # EmptySection has no content, should produce no chunk
        assert len(chunks) == 1
        assert "Skills" in chunks[0]

    def test_content_before_first_heading_is_intro(self):
        chunker = SectionChunker()
        text = "Introductory text before any heading.\n## Skills\nPython"
        chunks = chunker.chunk(text)
        intro_chunks = [c for c in chunks if "[Section: intro]" in c]
        assert len(intro_chunks) == 1

    def test_chunk_with_headings_returns_tuples(self):
        chunker = SectionChunker()
        text = "## Skills\nPython\n## Education\nCMU"
        result = chunker.chunk_with_headings(text)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        headings = [h for h, _ in result]
        assert "Skills" in headings
        assert "Education" in headings

    def test_empty_text_returns_empty_list(self):
        chunker = SectionChunker()
        assert chunker.chunk("") == []

    def test_single_heading_single_chunk(self):
        chunker = SectionChunker()
        text = "## About\nI am a data scientist."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert "[Section: About]" in chunks[0]
        assert "I am a data scientist." in chunks[0]
