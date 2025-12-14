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
        max_tokens: int = 512
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM server URL (e.g., "http://localhost:8000/v1")
            api_key: API key (use "EMPTY" for local vLLM)
            model: Model name to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        # Get from environment if not provided
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.model = model or os.getenv("VLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client pointing to vLLM
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key
        )
        
        print(f"Initialized vLLM client: {self.base_url}, model: {self.model}")
    
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
            
            return response.choices[0].text.strip()
        
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    
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
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    
    def generate_rag_response(
        self,
        query: str,
        chunks: List[str],
        max_chunks: int = 5
    ) -> str:
        """
        Generate RAG response from query and retrieved chunks.
        
        Args:
            query: User query
            chunks: Retrieved text chunks
            max_chunks: Maximum number of chunks to use
            
        Returns:
            Generated response
        """
        # Limit number of chunks
        chunks = chunks[:max_chunks]
        
        # Build context from chunks
        context = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])
        
        # Create prompt
        prompt = self._build_rag_prompt(query, context)
        
        # Generate using chat API for better instruction following
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        return self.generate_with_chat(messages)
    
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

Given the context information above, please answer the following query:
{query}

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
