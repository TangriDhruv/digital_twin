"""
github_loader.py
----------------
Fetches GitHub repos, commit history, and READMEs for a user
and converts them into chunk dicts ready for embedding.

Each non-fork repo produces up to 3 chunks:
  1. Repo overview  (name, description, language, topics, stars)
  2. Recent commits (grouped commit messages with dates)
  3. README content (chunked if long, skipped if missing)
"""

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from config import (
    GITHUB_API_BASE,
    GITHUB_MAX_COMMITS,
    GITHUB_MAX_REPOS,
    GITHUB_USERNAME,
)
from loaders import BaseLoader
from chunkers import ParagraphChunker

log = logging.getLogger(__name__)

_DUMMY_PATH = Path("github://live")


class GitHubLoader(BaseLoader):
    """
    Fetches public GitHub data for GITHUB_USERNAME.
    Plugs into the existing ingestion pipeline via BaseLoader.
    """

    def __init__(
        self,
        username: str = GITHUB_USERNAME,
        max_repos: int = GITHUB_MAX_REPOS,
        max_commits: int = GITHUB_MAX_COMMITS,
        api_base: str = GITHUB_API_BASE,
    ):
        self._username    = username
        self._max_repos   = max_repos
        self._max_commits = max_commits
        self._api_base    = api_base
        self._token       = os.getenv("GITHUB_TOKEN")
        self._chunker     = ParagraphChunker()   # reuse existing chunker for READMEs

    def load(self, path: Path = _DUMMY_PATH) -> list[dict]:
        log.info(f"Fetching GitHub data for @{self._username}...")
        chunks: list[dict] = []

        repos = self._fetch_repos()
        if not repos:
            log.warning("No repos fetched — check GITHUB_USERNAME in config.py")
            return chunks

        log.info(f"Found {len(repos)} repos — fetching commits + READMEs...")

        for repo in repos:
            if repo.get("fork"):
                continue

            # 1. Repo overview
            overview = self._make_repo_overview_chunk(repo)
            if overview:
                chunks.append(overview)

            # 2. Commits
            commits = self._fetch_commits(repo["name"])
            if commits:
                chunks.append(self._make_commit_chunk(repo, commits))

            # 3. README
            readme_chunks = self._fetch_and_chunk_readme(repo)
            chunks.extend(readme_chunks)

            time.sleep(0.15)   # stay within rate limits

        log.info(f"  → {len(chunks)} chunks from GitHub")
        return chunks

    # ── API helpers ────────────────────────────────────────────────────────

    def _fetch(self, url: str) -> list | dict | None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code != 404:   # 404 is expected for repos without README
                log.warning(f"GitHub API {e.code} → {url}")
            return None
        except Exception as e:
            log.warning(f"GitHub request failed: {e}")
            return None

    def _fetch_repos(self) -> list[dict]:
        url = (
            f"{self._api_base}/users/{self._username}/repos"
            f"?sort=updated&direction=desc"
            f"&per_page={self._max_repos}&type=public"
        )
        data = self._fetch(url)
        return data if isinstance(data, list) else []

    def _fetch_commits(self, repo_name: str) -> list[dict]:
        url = (
            f"{self._api_base}/repos/{self._username}/{repo_name}/commits"
            f"?per_page={self._max_commits}"
        )
        data = self._fetch(url)
        return data if isinstance(data, list) else []

    def _fetch_readme(self, repo_name: str) -> str | None:
        """
        Fetch and decode the README for a repo.
        GitHub returns content as base64 — we decode to plain text.
        Returns None if repo has no README.
        """
        url = f"{self._api_base}/repos/{self._username}/{repo_name}/readme"
        data = self._fetch(url)
        if not data or "content" not in data:
            return None

        try:
            # GitHub base64-encodes with newlines every 60 chars
            raw = data["content"].replace("\n", "")
            decoded = base64.b64decode(raw).decode("utf-8", errors="ignore")
            return decoded.strip() or None
        except Exception as e:
            log.warning(f"README decode failed for {repo_name}: {e}")
            return None

    # ── Chunk builders ─────────────────────────────────────────────────────

    def _make_repo_overview_chunk(self, repo: dict) -> dict | None:
        name        = repo.get("name", "")
        description = repo.get("description") or "No description provided"
        language    = repo.get("language") or "Unknown"
        stars       = repo.get("stargazers_count", 0)
        topics      = repo.get("topics", [])
        updated_at  = (repo.get("updated_at") or "")[:10]
        url         = repo.get("html_url", "")
        topics_str  = ", ".join(topics) if topics else "none"

        text = (
            f"[GitHub Repository: {name}]\n"
            f"Description: {description}\n"
            f"Primary language: {language}\n"
            f"Topics: {topics_str}\n"
            f"Stars: {stars}\n"
            f"Last updated: {updated_at}\n"
            f"URL: {url}"
        )

        return {
            "text":    text,
            "source":  "github",
            "topic":   "github_repos",
            "section": name,
            "type":    "structured",
            "repo":    name,
        }

    def _make_commit_chunk(self, repo: dict, commits: list[dict]) -> dict:
        repo_name = repo.get("name", "")
        lines = []
        for commit in commits:
            msg = (
                commit.get("commit", {})
                      .get("message", "")
                      .split("\n")[0]
                      .strip()
            )
            date = (
                commit.get("commit", {})
                      .get("author", {})
                      .get("date", "")[:10]
            )
            if msg:
                lines.append(f"  [{date}] {msg}")

        commits_text = "\n".join(lines) if lines else "  No commits found"

        return {
            "text": (
                f"[GitHub Commits: {repo_name}]\n"
                f"Recent commits in {repo_name}:\n"
                f"{commits_text}"
            ),
            "source":  "github",
            "topic":   "github_commits",
            "section": repo_name,
            "type":    "narrative",
            "repo":    repo_name,
        }

    def _fetch_and_chunk_readme(self, repo: dict) -> list[dict]:
        """
        Fetch README, strip markdown symbols, chunk it, and return
        a list of chunk dicts. Returns [] if no README exists.
        """
        repo_name = repo.get("name", "")
        raw = self._fetch_readme(repo_name)
        if not raw:
            return []

        # Light cleanup — remove markdown image tags and HTML tags
        # but keep headings and text so context is preserved
        cleaned = self._clean_markdown(raw)

        # Reuse ParagraphChunker — same logic as markdown docs
        text_chunks = self._chunker.chunk(cleaned)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            prefix = f"[GitHub README: {repo_name} | chunk {i+1}]\n"
            chunks.append({
                "text":    prefix + chunk_text,
                "source":  "github",
                "topic":   "github_readme",
                "section": repo_name,
                "type":    "narrative",
                "repo":    repo_name,
                "chunk_index": i,
            })

        log.info(f"  README: {repo_name} → {len(chunks)} chunks")
        return chunks

    def _clean_markdown(self, text: str) -> str:
        """
        Light markdown cleanup to improve embedding quality.
        Removes noise (image tags, badges, HTML) while keeping
        the actual content intact.
        """
        import re
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Remove markdown image syntax  ![alt](url)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        # Remove badge URLs (common in READMEs, add noise)
        text = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", text)
        # Remove raw URLs
        text = re.sub(r"https?://\S+", "", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()