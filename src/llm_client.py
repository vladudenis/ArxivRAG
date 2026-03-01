"""
DeepSeek API client for text generation.
Uses OpenAI-compatible API (DeepSeek API).
"""
import os
from typing import List, Dict
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
        chunks: List[str],
        max_chunks: int = 5,
        _retry_count: int = 0,
    ) -> str:
        """Generate RAG response from query and retrieved chunks."""
        chunks = chunks[:max_chunks]

        tokens_for_context = self.context_limit - self.max_tokens - 1000
        tokens_for_context = max(500, tokens_for_context)
        MAX_CONTEXT_CHARS = int(tokens_for_context * 3.5)

        format_overhead_per_chunk = 50
        available_for_content = MAX_CONTEXT_CHARS - (len(chunks) * format_overhead_per_chunk)
        chars_per_chunk = max(2000, available_for_content // len(chunks)) if chunks else 0

        truncated_chunks = []
        for chunk in chunks:
            if len(chunk) > chars_per_chunk:
                truncated_chunks.append(chunk[:chars_per_chunk] + "...")
            else:
                truncated_chunks.append(chunk)

        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(truncated_chunks)])

        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "...(truncated)"

        prompt = self._build_rag_prompt(query, context)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Provide specific, detailed answers with technical details such as model names, architectures, and methodologies when available in the context. Use only the information from the context to answer. When citing information from the context, use the source number in square brackets, e.g. [1] or [2]. Only cite sources you actually use.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            return self.generate_with_chat(messages)
        except RuntimeError as e:
            error_str = str(e)
            if "maximum context length" in error_str and _retry_count < 2 and max_chunks > 1:
                return self.generate_rag_response(
                    query, chunks, max_chunks=max(1, max_chunks // 2), _retry_count=_retry_count + 1
                )
            raise

    def _build_rag_prompt(self, query: str, context: str) -> str:
        """Build RAG prompt from query and context."""
        return f"""Context information is below:
---
{context}
---

Given the context information above, please answer the following query with specific details:
{query}

Provide a detailed answer that includes specific model names, architecture details, and technical specifications when available in the context. Be precise and cite specific information from the context using [1], [2], etc. for the source numbers.

Answer:"""


if __name__ == "__main__":
    client = DeepSeekClient()
    test_chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
    ]
    response = client.generate_rag_response("What is machine learning?", test_chunks)
    print(f"Response: {response}")
