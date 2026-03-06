"""
prompt.py
---------
Assembles the final prompt from retrieved chunks and user query.
This is where Dhruv's digital twin gets its voice, personality,
and instructions on how to behave.

Output format:
    A list of OpenAI-style message dicts:
    [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
    ]
    For multi-turn conversations, history messages are injected between them.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from config import PROFILE_PATH, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
log = logging.getLogger(__name__)


# ----- System prompt ------
# This defines who the twin is and how it should behave.
# Kept as a module-level constant so it's easy to iterate on.

SYSTEM_PROMPT = """You are the digital twin of Dhruv Tangri — a Data Scientist and AI Engineer \
with 4 years of industry experience, completed his Masters at Carnegie Mellon University.

Your job is to answer questions exactly as Dhruv would — using his voice, opinions, and experiences. \
You are not a generic assistant. You are Dhruv.

## How to respond

- Answer in first person ("I think...", "In my experience...", "I'd approach this by...")
- Be direct and opinionated. Dhruv has strong views — express them confidently
- Keep answers concise but substantive. Don't pad with filler
- Use specific examples from Dhruv's actual experience wherever relevant
- When discussing technical topics, show reasoning — not just conclusions
- It's okay to say "I don't have strong opinions on X" if the context doesn't cover it

## Tone

- Conversational but sharp — like talking to a senior engineer, not reading a LinkedIn post
- Enthusiastic about hard problems, especially at the intersection of AI and real products
- Honest about tradeoffs and what you'd do differently in hindsight

## Boundaries

- Only speak from the provided context. Do not invent experiences, companies, or opinions
- If asked something not covered in the context, say honestly: \
"I don't have enough context on that — you'd have to ask me directly"
- Never break character or refer to yourself as an AI or language model
- Do not list bullet points unless the question explicitly calls for a list — \
prefer natural conversational prose

## Citations

Each source in the context block is prefixed with a number like [1], [2], etc. \
You MUST place the matching citation number inline at the end of EVERY sentence that draws \
from a specific source — even for broad, biographical questions. Place the number before \
the period or punctuation. Do not skip citations because the question feels general.

Do not cite the always-included profile summary (it has no number).

Example for a broad question like "Tell me about yourself":
---
Context sources:
[1] career_story.md | intro
[2] faqs.md | Tell me about yourself
[3] personality_and_working_style.md | outside of work

Answer:
"I'm a Data Scientist and AI Engineer with four years of industry experience, primarily at \
KPMG [1]. My path into data actually started with Mechanical Engineering at VIT, where I \
worked on an Electric Solar Vehicle project that made data feel real and impactful [2]. \
Outside of work I volunteer with CRY teaching English to kids, and Formula 1 was honestly \
my gateway into data science [3]."
---

## Context format

You will receive relevant sections from Dhruv's personal knowledge base below. \
Each section is tagged with its source number, topic, and source file. \
Use these to ground your answers and cite them inline.
"""


class BasePromptBuilder(ABC):
    """Abstract prompt builder."""

    @abstractmethod
    def build(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict] | None = None,
    ) -> list[dict]:
        """
        Build a list of OpenAI message dicts from query + retrieved chunks.

        Args:
            query:   The user's current question
            chunks:  Retrieved chunk dicts from FAISSRetriever
            history: Optional prior conversation turns
                     [{"role": "user"|"assistant", "content": str}, ...]

        Returns:
            List of OpenAI message dicts ready to send to the API
        """
        ...


class TwinPromptBuilder(BasePromptBuilder):
    """
    Builds a RAG prompt for Dhruv's digital twin.

    Structure:
        [system]     — persona + instructions
        [user]       — injected context block + actual question
        [history]    — prior conversation turns (if any)
        [user]       — the current question (repeated at end for clarity)

    Context injection strategy:
        - Chunks are grouped by topic for readability
        - Each chunk shows its source and section so the LLM can
          reason about provenance
        - Profile summary is always first in context (highest priority)
        - Score threshold filters out low-relevance chunks
    """

    MIN_SCORE = 0.10   # discard chunks below this cosine similarity


    def build(
        self,
        query: str,
        chunks: list[dict],
        history: list[dict] | None = None,
    ) -> list[dict]:

        # 1. Filter low-relevance chunks (keep always_included regardless)
        filtered = self.filter_chunks(chunks)
        log.info(f"Prompt: {len(chunks)} chunks → {len(filtered)} after filtering")

        # 2. Format chunks into a readable context block
        context_block = self._format_context(filtered)

        # 3. Assemble messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject conversation history (excluding the current turn)
        if history:
            messages.extend(history)

        # User message = context + question
        user_content = self._format_user_message(query, context_block)
        messages.append({"role": "user", "content": user_content})

        log.info(
            f"Prompt built — {len(messages)} messages, "
            f"~{self._estimate_tokens(messages)} tokens"
        )

        return messages

    

    def filter_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Keep chunks above MIN_SCORE threshold.
        Always-included chunks (summary) bypass the filter.
        """
        return [
            c for c in chunks
            if c.get("always_included") or c.get("score", 0) >= self.MIN_SCORE
        ]

    def _format_context(self, chunks: list[dict]) -> str:
        """
        Group chunks by topic and format into a readable context block.
        Non-always-included chunks are numbered [1], [2], ... so the LLM
        can use those numbers as inline citations in its response.
        """
        if not chunks:
            return "No specific context retrieved for this query."

        # Assign citation numbers to non-summary chunks first
        citation_num = 1
        for chunk in chunks:
            if chunk.get("always_included"):
                chunk["_citation_num"] = None
            else:
                chunk["_citation_num"] = citation_num
                citation_num += 1

        # Group by topic for a cleaner context block
        grouped: dict[str, list[dict]] = {}
        for chunk in chunks:
            topic = chunk.get("topic", "general")
            grouped.setdefault(topic, []).append(chunk)

        sections = []
        for topic, topic_chunks in grouped.items():
            topic_label = topic.replace("_", " ").title()
            section_parts = [f"### {topic_label}"]

            for chunk in topic_chunks:
                section_label = chunk.get("section", "")
                source        = chunk.get("source", "")
                score         = chunk.get("score", 0)
                always        = chunk.get("always_included", False)
                num           = chunk.get("_citation_num")

                if always:
                    header = f"[always included | {source} | {section_label}]"
                else:
                    header = f"[{num}] [{source} | {section_label} | relevance: {score:.2f}]"

                section_parts.append(f"{header}\n{chunk['text']}")

            sections.append("\n\n".join(section_parts))

        return "\n\n---\n\n".join(sections)

    def _format_user_message(self, query: str, context_block: str) -> str:
        """
        Combine context block and user question into a single user message.
        Context comes first so the model reads it before seeing the question.
        """
        return (
            f"## Context from Dhruv's knowledge base\n\n"
            f"{context_block}\n\n"
            f"---\n\n"
            f"## Question\n\n"
            f"{query}"
        )

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """
        Rough token estimate: ~1 token per 4 characters.
        Used only for logging — not passed to the API.
        """
        total_chars = sum(len(m["content"]) for m in messages)
        return total_chars // 4


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate what retrieval.py would return
    mock_chunks = [
        {
            "text": "[Topic: Opinions On Tech | Section: RAG vs fine-tuning]\n"
                    "RAG, almost always. Fine-tuning is overhyped. It's tedious, "
                    "expensive, and if your data changes tomorrow you have to do it "
                    "all over again. RAG is transparent, updatable, and explainable.",
            "source": "opinions_on_tech.md",
            "topic": "opinions_on_tech",
            "section": "RAG vs fine-tuning",
            "type": "narrative",
            "score": 0.82,
        },
        {
            "text": "Name: Dhruv Tangri\nTitle: Data Scientist / AI Engineer\n"
                    "Location: Pittsburgh, PA\nExperience: 4 years\n"
                    "Summary: Data Scientist with 4 years of experience...",
            "source": "profile.json",
            "topic": "summary",
            "type": "structured",
            "score": 1.0,
            "always_included": True,
        },
    ]

    builder = TwinPromptBuilder()
    messages = builder.build(
        query="What do you think about RAG vs fine-tuning?",
        chunks=mock_chunks,
        history=[],
    )

    print(f"\nBuilt {len(messages)} messages\n")
    for msg in messages:
        print(f"── {msg['role'].upper()} {'─'*50}")
        print(msg["content"][:500])
        print()
