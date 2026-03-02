"""
test_prompt.py
--------------
Unit tests for TwinPromptBuilder.
No OpenAI or FAISS calls — all inputs are mocked chunk dicts.
"""

import pytest
from prompt import TwinPromptBuilder


@pytest.fixture
def builder():
    return TwinPromptBuilder()


@pytest.fixture
def summary_chunk():
    """Profile summary chunk — always_included=True."""
    return {
        "text": "Name: Dhruv Tangri. Data Scientist with 4 years of experience.",
        "source": "profile.json",
        "topic": "summary",
        "section": "summary",
        "score": 1.0,
        "always_included": True,
    }


@pytest.fixture
def high_score_chunk():
    return {
        "text": "RAG is almost always better than fine-tuning.",
        "source": "opinions_on_tech.md",
        "topic": "opinions_on_tech",
        "section": "RAG vs fine-tuning",
        "score": 0.82,
    }


@pytest.fixture
def low_score_chunk():
    return {
        "text": "Barely relevant content.",
        "source": "docs.md",
        "topic": "misc",
        "section": "other",
        "score": 0.05,  # Below MIN_SCORE of 0.10
    }


# ── Message structure ─────────────────────────────────────────────────────────

class TestMessageStructure:

    def test_first_message_is_system(self, builder, summary_chunk):
        messages = builder.build("Who are you?", [summary_chunk])
        assert messages[0]["role"] == "system"

    def test_last_message_is_user(self, builder, summary_chunk):
        messages = builder.build("Who are you?", [summary_chunk])
        assert messages[-1]["role"] == "user"

    def test_system_prompt_contains_dhruv(self, builder, summary_chunk):
        messages = builder.build("Who are you?", [summary_chunk])
        assert "Dhruv" in messages[0]["content"]

    def test_user_message_contains_query(self, builder, summary_chunk):
        query = "What is your favourite programming language?"
        messages = builder.build(query, [summary_chunk])
        assert query in messages[-1]["content"]

    def test_user_message_contains_context(self, builder, summary_chunk):
        messages = builder.build("Who are you?", [summary_chunk])
        assert "Context" in messages[-1]["content"]


# ── Chunk filtering ───────────────────────────────────────────────────────────

class TestChunkFiltering:

    def test_always_included_chunk_bypasses_score_filter(
        self, builder, summary_chunk, low_score_chunk
    ):
        messages = builder.build("Query", [summary_chunk, low_score_chunk])
        user_content = messages[-1]["content"]
        # summary chunk (always_included) must appear
        assert "Dhruv Tangri" in user_content

    def test_low_score_chunk_is_filtered_out(
        self, builder, summary_chunk, low_score_chunk
    ):
        messages = builder.build("Query", [summary_chunk, low_score_chunk])
        user_content = messages[-1]["content"]
        assert "Barely relevant content." not in user_content

    def test_high_score_chunk_is_included(
        self, builder, summary_chunk, high_score_chunk
    ):
        messages = builder.build("Query", [summary_chunk, high_score_chunk])
        user_content = messages[-1]["content"]
        assert "RAG is almost always better" in user_content

    def test_empty_chunks_returns_fallback_context(self, builder):
        messages = builder.build("Who are you?", [])
        user_content = messages[-1]["content"]
        assert "No specific context retrieved" in user_content


# ── Conversation history ──────────────────────────────────────────────────────

class TestHistory:

    def test_history_injected_between_system_and_user(
        self, builder, summary_chunk
    ):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        messages = builder.build("Follow-up question", [summary_chunk], history=history)
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert roles[1] == "user"      # history turn 1
        assert roles[2] == "assistant" # history turn 2
        assert roles[3] == "user"      # current question

    def test_no_history_produces_two_messages(self, builder, summary_chunk):
        messages = builder.build("Question", [summary_chunk], history=None)
        assert len(messages) == 2  # system + user

    def test_empty_history_produces_two_messages(self, builder, summary_chunk):
        messages = builder.build("Question", [summary_chunk], history=[])
        assert len(messages) == 2


# ── Context formatting ────────────────────────────────────────────────────────

class TestContextFormatting:

    def test_context_groups_by_topic(self, builder, summary_chunk, high_score_chunk):
        messages = builder.build("Query", [summary_chunk, high_score_chunk])
        user_content = messages[-1]["content"]
        # Topic headings should appear
        assert "Summary" in user_content or "Opinions On Tech" in user_content

    def test_token_estimate_is_positive(self, builder, summary_chunk):
        messages = builder.build("What is your experience?", [summary_chunk])
        estimate = builder._estimate_tokens(messages)
        assert estimate > 0
