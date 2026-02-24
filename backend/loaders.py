"""
loaders.py
----------
File loaders that read raw data and return structured chunk dicts.



Each chunk dict has at minimum:
    {
        "text":   str,          # the actual text to embed
        "source": str,          # filename
        "topic":  str,          # high-level topic label
        "type":   str,          # "structured" | "narrative"
    }
Plus optional fields: section, chunk_index, company, project, etc.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from chunkers import BaseChunker, SectionChunker

log = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract loader — all loaders must implement load()."""

    @abstractmethod
    def load(self, path: Path) -> list[dict]:
        """Read a file and return a list of chunk dicts ready for embedding."""
        ...


class MarkdownLoader(BaseLoader):
    """
    Loads a markdown file and splits it into chunks using SectionChunker.
    Each chunk carries metadata about its source file, topic, and section.
    A context prefix is injected into every chunk text so the embedding
    captures where the content comes from, not just what it says.
    """

    def __init__(self, chunker: BaseChunker | None = None):
        # D: accepts any BaseChunker — defaults to SectionChunker
        self._chunker = chunker or SectionChunker()

    def load(self, path: Path) -> list[dict]:
        log.info(f"Loading markdown: {path.name}")
        text = path.read_text(encoding="utf-8")
        topic = path.stem  # filename without extension used as topic

        # SectionChunker exposes chunk_with_headings for richer metadata
        if isinstance(self._chunker, SectionChunker):
            raw_chunks = self._chunker.chunk_with_headings(text)
        else:
            raw_chunks = [(None, c) for c in self._chunker.chunk(text)]

        chunks: list[dict] = []
        for i, (heading, chunk_text) in enumerate(raw_chunks):
            topic_label = topic.replace("_", " ").title()
            section_label = heading or "General"

            # Context prefix makes retrieval topic-aware
            prefixed_text = (
                f"[Topic: {topic_label} | Section: {section_label}]\n{chunk_text}"
            )

            chunks.append({
                "text":        prefixed_text,
                "source":      path.name,
                "topic":       topic,
                "section":     section_label,
                "type":        "narrative",
                "chunk_index": i,
            })

        log.info(f"  → {len(chunks)} chunks from {path.name}")
        return chunks


class JSONProfileLoader(BaseLoader):
    """
    Loads profile.json and converts each logical section into its own chunk.
    Structured data is never mixed — skills, education, each role, each project
    all become separate chunks for precise retrieval.

    No chunker needed here: JSON sections are naturally bounded.
    """

    def load(self, path: Path) -> list[dict]:
        log.info(f"Loading profile: {path.name}")
        with open(path, encoding="utf-8") as f:
            profile = json.load(f)

        chunks: list[dict] = []
        chunks.extend(self._summary_chunk(profile, path.name))
        chunks.extend(self._skills_chunk(profile, path.name))
        chunks.extend(self._education_chunks(profile, path.name))
        chunks.extend(self._work_history_chunks(profile, path.name))
        chunks.extend(self._project_chunks(profile, path.name))
        chunks.extend(self._awards_chunk(profile, path.name))

        log.info(f"  → {len(chunks)} chunks from {path.name}")
        return chunks

    # ── Private section builders ───────────────────────────────────────────

    def _make_chunk(self, text: str, source: str, topic: str, **extra) -> dict:
        return {"text": text, "source": source, "topic": topic, "type": "structured", **extra}

    def _summary_chunk(self, p: dict, source: str) -> list[dict]:
        text = (
            f"Name: {p['name']}\n"
            f"Title: {p['title']}\n"
            f"Location: {p['location']}\n"
            f"Experience: {p['experience_years']} years\n"
            f"Summary: {p['summary']}\n"
            f"Looking for: {p['looking_for']}\n"
            f"Open to: {', '.join(p['open_to'])}\n"
            f"Not interested in: {', '.join(p['not_interested_in'])}"
        )
        return [self._make_chunk(text, source, "summary")]

    def _skills_chunk(self, p: dict, source: str) -> list[dict]:
        s = p["skills"]
        text = (
            f"Technical Skills:\n"
            f"Languages: {', '.join(s['languages'])}\n"
            f"Frameworks: {', '.join(s['frameworks'])}\n"
            f"Tools: {', '.join(s['tools'])}\n"
            f"Databases: {', '.join(s['databases'])}\n"
            f"Cloud: {', '.join(s.get('cloud', []))}\n"
            f"ML/AI: {', '.join(s.get('ml', []))}"
        )
        return [self._make_chunk(text, source, "skills")]

    def _education_chunks(self, p: dict, source: str) -> list[dict]:
        lines = []
        for edu in p["education"]:
            lines.append(
                f"{edu['degree']} — {edu['school']} ({edu['year']}) | GPA: {edu['gpa']}\n"
                f"Notes: {edu['notes']}"
            )
        text = "Education:\n" + "\n\n".join(lines)
        return [self._make_chunk(text, source, "education")]

    def _work_history_chunks(self, p: dict, source: str) -> list[dict]:
        chunks = []
        for job in p["work_history"]:
            text = (
                f"Role: {job['role']} at {job['company']} "
                f"({job['years']}) — {job['location']}\n"
                f"Summary: {job['summary']}\n"
                f"Impact: {job['impact']}"
            )
            chunks.append(
                self._make_chunk(text, source, "work_history", company=job["company"])
            )
        return chunks

    def _project_chunks(self, p: dict, source: str) -> list[dict]:
        chunks = []
        for proj in p["side_projects"]:
            text = (
                f"Project: {proj['name']} ({proj.get('year', '')})\n"
                f"Description: {proj['description']}\n"
                f"Tech stack: {', '.join(proj['tech_stack'])}\n"
                f"Impact: {proj.get('impact', 'N/A')}"
            )
            chunks.append(
                self._make_chunk(text, source, "projects", project=proj["name"])
            )
        return chunks

    def _awards_chunk(self, p: dict, source: str) -> list[dict]:
        text = "Awards & Recognition:\n" + "\n".join(
            f"- {a}" for a in p.get("awards", [])
        )
        return [self._make_chunk(text, source, "awards")]
