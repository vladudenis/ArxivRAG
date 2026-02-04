"""
vLLM client for text generation.
Supports OpenAI-compatible API interface.
"""
import os
from typing import List, Dict, Optional
from openai import OpenAI

class VLLMClient:
    """Client for vLLM server with OpenAI-compatible API."""
    
    def __init__(
        self, 
        base_url: str = None,
        api_key: str = "EMPTY",
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        context_limit: int = None
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM server URL (e.g., "http://localhost:8000/v1")
            api_key: API key (use "EMPTY" for local vLLM)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            context_limit: Maximum context window in tokens (auto-detected if None)
        """
        # Get from environment if not provided
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        # Default to TinyLlama for pipeline (can be overridden via VLLM_MODEL env var)
        # Interactive RAG uses DeepSeek API (configured separately)
        self.model = model or os.getenv("VLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Detect context limit based on model name if not provided
        if context_limit is None:
            self.context_limit = self._detect_context_limit(self.model)
        else:
            self.context_limit = context_limit
        
        # Initialize OpenAI client pointing to vLLM
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key
        )
        
        print(f"Initialized vLLM client: {self.base_url}, model: {self.model}, context_limit: {self.context_limit} tokens")
    
    def _detect_context_limit(self, model_name: str) -> int:
        """
        Detect context limit based on model name.
        
        Args:
            model_name: Model name
            
        Returns:
            Context limit in tokens
        """
        model_lower = model_name.lower()
        
        # DeepSeek models
        if 'deepseek' in model_lower:
            return 8192  # DeepSeek has 8k context
        
        # Models with 8k+ context
        if any(x in model_lower for x in ['qwen2', 'qwen-2', 'mistral-7b', 'mistral-8x7b', 'llama-3']):
            return 8192
        
        # Models with 4k context
        if any(x in model_lower for x in ['llama-2', 'llama2', 'neural-chat']):
            return 4096
        
        # Default to 2048 for smaller models (TinyLlama, etc.)
        return 2048
    
    def generate(
        self, 
        prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text
        """
        try:
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            text = response.choices[0].text
            if text:
                return text.strip()
            else:
                print("Warning: Empty response from LLM")
                return ""
        
        except Exception as e:
            import traceback
            error_msg = f"Error generating text: {e}\n{traceback.format_exc()}"
            print(error_msg)
            raise RuntimeError(f"Failed to generate text: {e}") from e
    
    def check_health(self) -> bool:
        """
        Check if vLLM server is available and responding.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Try a simple completion request
            response = self.client.completions.create(
                model=self.model,
                prompt="test",
                max_tokens=1
            )
            return True
        except Exception as e:
            print(f"vLLM health check failed: {e}")
            return False
    
    def generate_with_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate text using chat completion API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            content = response.choices[0].message.content
            if content:
                return content.strip()
            else:
                print("Warning: Empty response from LLM")
                return ""
        
        except Exception as e:
            import traceback
            error_msg = f"Error generating text: {e}\n{traceback.format_exc()}"
            print(error_msg)
            raise RuntimeError(f"Failed to generate text: {e}") from e
    
    def generate_rag_response(
        self,
        query: str,
        chunks: List[str],
        max_chunks: int = 5,
        _retry_count: int = 0
    ) -> str:
        """
        Generate RAG response from query and retrieved chunks.
        Automatically adjusts context size based on model's context limit.
        
        Args:
            query: User query
            chunks: Retrieved text chunks
            max_chunks: Maximum number of chunks to use
            
        Returns:
            Generated response
        """
        # Limit number of chunks
        chunks = chunks[:max_chunks]
        
        # Calculate max context chars based on context limit
        # Leave room for: generation tokens + system/user prompts (~500-1000 tokens)
        # Approx 3-4 chars per token
        tokens_for_context = self.context_limit - self.max_tokens - 1000  # Reserve 1000 for prompts
        tokens_for_context = max(500, tokens_for_context)  # At least 500 tokens
        MAX_CONTEXT_CHARS = int(tokens_for_context * 3.5)  # Conservative: 3.5 chars per token
        
        # Truncate each chunk proportionally to preserve information from all chunks
        # Reserve ~50 chars per chunk for formatting ("[1] ", "\n\n", etc.)
        format_overhead_per_chunk = 50
        available_for_content = MAX_CONTEXT_CHARS - (len(chunks) * format_overhead_per_chunk)
        # Calculate chars per chunk - more for larger context windows
        if self.context_limit >= 8192:
            chars_per_chunk = max(2000, available_for_content // len(chunks)) if chunks else 0
        elif self.context_limit >= 4096:
            chars_per_chunk = max(1000, available_for_content // len(chunks)) if chunks else 0
        else:  # 2048 token models
            chars_per_chunk = max(300, available_for_content // len(chunks)) if chunks else 0
        
        truncated_chunks = []
        for chunk in chunks:
            if len(chunk) > chars_per_chunk:
                truncated_chunks.append(chunk[:chars_per_chunk] + "...")
            else:
                truncated_chunks.append(chunk)
        
        # Build context from truncated chunks
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(truncated_chunks)])
        
        # Final safety check - truncate if still too long
        if len(context) > MAX_CONTEXT_CHARS:
            print(f"Warning: Context still too long ({len(context)} chars), truncating to {MAX_CONTEXT_CHARS}...")
            context = context[:MAX_CONTEXT_CHARS] + "...(truncated)"
        
        # Create prompt
        prompt = self._build_rag_prompt(query, context)
        
        # Generate using chat API for better instruction following
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Provide specific, detailed answers with technical details such as model names, architectures, and methodologies when available in the context. If the context mentions specific models (e.g., BERT, BART, GPT, DPR), architectures (e.g., bi-encoder, encoder-decoder), or technical details, include them in your answer. Use only the information from the context to answer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Try generation, with automatic retry on context limit errors
        try:
            return self.generate_with_chat(messages)
        except RuntimeError as e:
            error_str = str(e)
            # If context too long error and we haven't retried too many times, retry with smaller context
            if "maximum context length" in error_str and _retry_count < 2 and max_chunks > 1:
                print(f"Context limit exceeded, retrying with reduced chunks ({max_chunks} -> {max(1, max_chunks // 2)})...")
                # Retry with fewer chunks
                return self.generate_rag_response(query, chunks, max_chunks=max(1, max_chunks // 2), _retry_count=_retry_count + 1)
            raise
    
    def _build_rag_prompt(self, query: str, context: str) -> str:
        """
        Build RAG prompt from query and context.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Context information is below:
---
{context}
---

Given the context information above, please answer the following query with specific details:
{query}

Provide a detailed answer that includes:
- Specific model names (e.g., BERT, BART, GPT, DPR, T5) if mentioned
- Architecture details (e.g., bi-encoder, encoder-decoder, transformer) if mentioned
- Technical specifications and methodologies if available
- Be precise and cite specific information from the context

Answer:"""
        
        return prompt

# Test function
if __name__ == "__main__":
    # Test vLLM connection
    client = VLLMClient()
    
    test_chunks = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language."
    ]
    
    test_query = "What is machine learning?"
    
    print("Testing vLLM client...")
    response = client.generate_rag_response(test_query, test_chunks)
    print(f"Query: {test_query}")
    print(f"Response: {response}")
