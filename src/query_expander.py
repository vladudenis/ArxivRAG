"""
Query expansion using local vLLM (OpenAI-compatible API).
"""
from __future__ import annotations

import os
import re

from openai import OpenAI


class QueryExpander:
    """Expand user query into keywords for arXiv search using local vLLM."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        """
        Initialize QueryExpander.

        Args:
            base_url: vLLM API base URL (from QUERY_EXPANSION_BASE_URL env).
            model: Model name (from QUERY_EXPANSION_MODEL env).
        """
        self.base_url = base_url or os.getenv("QUERY_EXPANSION_BASE_URL", "http://localhost:8001/v1")
        self.model = model or os.getenv("QUERY_EXPANSION_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI | None:
        """Get or create OpenAI client for vLLM."""
        if self._client is None:
            try:
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=os.getenv("QUERY_EXPANSION_API_KEY", "dummy"),
                    timeout=60.0,
                )
            except Exception:
                return None
        return self._client

    def expand(self, query: str) -> str:
        """
        Expand query into 3-5 keywords for academic paper search.

        Args:
            query: User's natural language query.

        Returns:
            Space-separated keywords. Falls back to original query on failure.
        """
        client = self._get_client()
        if not client:
            return query

        # Minimal prompt - long instructions cause TinyLlama to paraphrase instead of comply
        prompt = f"Extract 3-6 technical search keywords from this query. \
                    Do not repeat instructions. \
                    Do not explain. \
                    Only output comma-separated keywords. \
                    Query: {query} \
                    Keywords:"

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0,
                top_p=0.9,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                return query

            keywords = self._extract_keywords(content)
            if keywords and self._is_valid_keywords(keywords, content):
                return keywords
        except Exception as e:
            print(f"  Query expansion failed ({e}), using original query")

        return query

    def _extract_keywords(self, content: str) -> str:
        """
        Extract space-separated keywords from model output, stripping verbose preamble.
        """
        text = content.strip()
        if not text:
            return ""

        # Common verbose prefixes to strip (case-insensitive)
        prefixes = [
            r"here\s+is\s+the\s+expanded\s+query\s*(?:for\s+the\s+input)?\s*[:\-]?\s*",
            r"here\s+are\s+the\s+(?:expanded\s+)?keywords?\s*[:\-]?\s*",
            r"the\s+(?:expanded\s+)?keywords?\s+(?:are|is)\s*[:\-]?\s*",
            r"expanded\s+query\s*[:\-]?\s*",
            r"keywords?\s*[:\-]?\s*",
        ]
        for pat in prefixes:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)

        # If model put keywords after a colon, take content after last colon
        if ":" in text and len(text) > 20:
            after_colon = text.rsplit(":", 1)[-1].strip()
            if len(after_colon) > 5 and " " in after_colon:
                text = after_colon

        # Take last line if multiple (model often puts keywords last)
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) > 1:
            for line in reversed(lines):
                if len(line) > 10 and " " in line and not line.endswith("."):
                    text = line
                    break
            else:
                text = lines[-1]
        elif lines:
            text = lines[0]

        # Normalize: collapse whitespace, remove trailing punctuation
        text = re.sub(r"\s+", " ", text.strip()).strip()
        text = re.sub(r"[.:;,!]+$", "", text)

        return text if text else ""

    def _is_valid_keywords(self, keywords: str, raw_content: str) -> bool:
        """
        Return False if output looks like instruction paraphrasing or garbage.
        """
        if len(keywords) < 4:
            return False

        # Phrases that indicate model paraphrased instructions or answered instead of keywords
        bad_phrases = [
            "output will",
            "only contain",
            "space-separated",
            "no other",
            "no other text",
            "expanded query",
            "keywords for",
            "the keywords",
            "output only",
            "just the",
            "with no",
            "and no",
            "preamble",
            "explanation",
            "the difference",
            "the key ",
            "the main ",
            "the following",
            "in summary",
            "in the context",
            "this is ",
            "this shows",
        ]
        lower = keywords.lower()
        if any(phrase in lower for phrase in bad_phrases):
            return False

        # Reject sentence starts (model answering instead of listing keywords)
        sentence_starts = ("the ", "in ", "this ", "these ", "that ", "it ")
        if lower.startswith(sentence_starts) and len(keywords.split()) <= 3:
            return False

        # Prefer multiple words; allow single word if substantial (e.g. "xLSTM")
        words = keywords.split()
        if len(words) < 2 and len(keywords) < 4:
            return False

        return True
