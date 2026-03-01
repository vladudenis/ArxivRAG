"""
DeepSeek API client for text generation.
Uses OpenAI-compatible API (DeepSeek API).
"""
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI


class DeepSeekClient:
    """Client for DeepSeek API (OpenAI-compatible)."""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        context_limit: int = 8192,
    ):
        """
        Initialize DeepSeek API client.

        Args:
            base_url: API base URL (from LLM_BASE_URL env)
            api_key: API key (from LLM_API_KEY env)
            model: Model name (default: deepseek-chat)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            context_limit: Context window in tokens (DeepSeek: 8192)
        """
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_limit = context_limit

        if not self.base_url or not self.api_key:
            raise ValueError("LLM_BASE_URL and LLM_API_KEY must be set in .env")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        print(f"Initialized DeepSeek client: {self.base_url}, model: {self.model}, context: {self.context_limit} tokens")

    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Generate text from prompt."""
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            text = response.choices[0].text
            return text.strip() if text else ""
        except Exception as e:
            import traceback
            print(f"Error generating text: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to generate text: {e}") from e

    def check_health(self) -> bool:
        """Check if DeepSeek API is available."""
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception as e:
            print(f"DeepSeek API health check failed: {e}")
            return False

    def generate_with_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """Generate text using chat completion API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            import traceback
            print(f"Error generating text: {e}\n{traceback.format_exc()}")
            raise RuntimeError(f"Failed to generate text: {e}") from e

    def generate_rag_response(
        self,
        query: str,
        chunks_with_meta: List[Tuple[str, Dict[str, Any]]],
        max_chunks: int = 5,
        _retry_count: int = 0,
    ) -> str:
        """Generate RAG response from query and retrieved chunks with metadata."""
        chunks_with_meta = chunks_with_meta[:max_chunks]

        tokens_for_context = self.context_limit - self.max_tokens - 1000
        tokens_for_context = max(500, tokens_for_context)
        MAX_CONTEXT_CHARS = int(tokens_for_context * 3.5)

        format_overhead_per_chunk = 80
        available_for_content = MAX_CONTEXT_CHARS - (len(chunks_with_meta) * format_overhead_per_chunk)
        chars_per_chunk = max(2000, available_for_content // len(chunks_with_meta)) if chunks_with_meta else 0

        context_parts: List[str] = []
        for chunk_text, payload in chunks_with_meta:
            title = payload.get("title", "Unknown")
            paper_id = payload.get("paper_id", "")
            section = payload.get("section") or "—"
            source_line = f"[Source: {title} (arxiv.org/abs/{paper_id}), Section: {section}]"
            if len(chunk_text) > chars_per_chunk:
                chunk_text = chunk_text[:chars_per_chunk] + "..."
            context_parts.append(f"{source_line}\n{chunk_text}")

        context = "\n\n".join(context_parts)

        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "...(truncated)"

        prompt = self._build_rag_prompt(query, context)

        messages = [
            {
                "role": "system",
                "content": "You are an expert who answers questions directly and authoritatively. Do not start with phrases like 'Based on the provided context' or 'According to the context'—answer as if you know the material. Use markdown for formatting: **bold** for terms and concepts, lists for enumerations. Be specific: include model names, architectures, and methodologies when available. Use only information from the context. When referencing a source, mention the paper title naturally (e.g. 'In the RAG survey by Smith et al.'). Do not use numeric indices like [1] or [2].",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            return self.generate_with_chat(messages)
        except RuntimeError as e:
            error_str = str(e)
            if "maximum context length" in error_str and _retry_count < 2 and max_chunks > 1:
                return self.generate_rag_response(
                    query, chunks_with_meta, max_chunks=max(1, max_chunks // 2), _retry_count=_retry_count + 1
                )
            raise

    def _build_rag_prompt(self, query: str, context: str) -> str:
        """Build RAG prompt from query and context."""
        return f"""Context:
---
{context}
---

Answer this question directly and in detail. Use markdown (**bold** for key terms). Include model names, architectures, and technical details when available. Reference papers by their title when citing.

Question: {query}

Answer:"""


if __name__ == "__main__":
    client = DeepSeekClient()
    test_chunks = [
        ("Machine learning is a subset of artificial intelligence.", {"title": "ML Basics", "paper_id": "1234.5678", "section": "Intro"}),
        ("Deep learning uses neural networks with multiple layers.", {"title": "DL Survey", "paper_id": "2345.6789", "section": "Background"}),
    ]
    response = client.generate_rag_response("What is machine learning?", test_chunks)
    print(f"Response: {response}")
