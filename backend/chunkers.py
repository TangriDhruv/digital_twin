"""
chunkers.py
-----------
Chunking strategies for splitting text into embeddable pieces.
"""

from abc import ABC, abstractmethod
from config import CHUNK_SIZE, CHUNK_OVERLAP


class BaseChunker(ABC):
    """Abstract base — all chunkers must implement chunk()."""

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into a list of string chunks."""
        ...


class ParagraphChunker(BaseChunker):
    """
    Chunks text by paragraph boundaries first, then by word count.
    Preserves semantic units (paragraphs) before falling back to
    hard word-count splits. Adds configurable overlap between chunks
    so context is not lost at boundaries.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks: list[str] = []
        current_words: list[str] = []

        for para in paragraphs:
            para_words = para.split()

            if (
                len(current_words) + len(para_words) > self.chunk_size
                and current_words
            ):
                chunks.append(" ".join(current_words))
                # Retain overlap window from previous chunk for context continuity
                current_words = current_words[-self.overlap:] if self.overlap else []

            current_words.extend(para_words)

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks


class SectionChunker(BaseChunker):
    """
    Splits markdown text on ## headings into named sections,
    then applies ParagraphChunker within each section.
    Each chunk is prefixed with its section heading for context.

    Designed specifically for structured markdown documents
    where ## headings define logical semantic units.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ):
        self._paragraph_chunker = ParagraphChunker(chunk_size, overlap)

    def chunk(self, text: str) -> list[str]:
        """Returns flat list of chunks across all sections."""
        sections = self._parse_sections(text)
        chunks: list[str] = []

        for heading, content in sections:
            if not content.strip():
                continue
            prefix = f"[Section: {heading}]\n"
            for sub_chunk in self._paragraph_chunker.chunk(content):
                chunks.append(prefix + sub_chunk)

        return chunks

    def chunk_with_headings(self, text: str) -> list[tuple[str, str]]:
        """
        Returns (heading, chunk_text) tuples.
        Useful when callers need to know which section a chunk came from.
        """
        sections = self._parse_sections(text)
        result: list[tuple[str, str]] = []

        for heading, content in sections:
            if not content.strip():
                continue
            for sub_chunk in self._paragraph_chunker.chunk(content):
                result.append((heading, sub_chunk))

        return result

    def _parse_sections(self, text: str) -> list[tuple[str, str]]:
        """Parse markdown into (heading, content) pairs on ## boundaries."""
        sections: list[tuple[str, str]] = []
        current_heading = "intro"
        current_lines: list[str] = []

        for line in text.split("\n"):
            if line.startswith("## "):
                if current_lines:
                    sections.append(
                        (current_heading, "\n".join(current_lines).strip())
                    )
                current_heading = line[3:].strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "\n".join(current_lines).strip()))

        return sections
