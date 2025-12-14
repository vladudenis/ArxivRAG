import fitz  # pymupdf
import re
import tiktoken
import nltk
from typing import List

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class Chunker:
    def __init__(self, strategy="recursive", chunk_size=500, chunk_overlap=50):
        """
        Initialize chunker with specified strategy.
        
        Args:
            strategy: One of 'fixed', 'recursive', 'paragraph', 'token', 'sentence'
            chunk_size: Target size for chunks (chars for most, tokens for 'token' strategy)
            chunk_overlap: Overlap size between chunks
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer for token-based chunking
        if strategy == "token":
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            except Exception as e:
                print(f"Warning: Could not load tiktoken encoding: {e}")
                print("Falling back to approximate token counting")
                self.tokenizer = None

    def extract_text_from_pdf(self, pdf_bytes):
        """Extracts text from PDF bytes."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def chunk_text(self, text):
        """Chunks text based on the selected strategy."""
        if not text:
            return []
            
        if self.strategy == "fixed":
            return self._fixed_size_chunking(text)
        elif self.strategy == "recursive":
            return self._recursive_chunking(text)
        elif self.strategy == "paragraph":
            return self._paragraph_chunking(text)
        elif self.strategy == "token":
            return self._token_based_chunking(text)
        elif self.strategy == "sentence":
            return self._sentence_based_chunking(text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _fixed_size_chunking(self, text):
        """Fixed-size character chunking with overlap."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
            
            # Prevent infinite loop
            if self.chunk_size <= self.chunk_overlap:
                break
            
        return chunks

    def _token_based_chunking(self, text: str) -> List[str]:
        """
        Token-based chunking using tiktoken.
        Chunks based on token count rather than character count.
        """
        if self.tokenizer is None:
            # Fallback: approximate 1 token ≈ 4 characters
            approx_char_size = self.chunk_size * 4
            approx_overlap = self.chunk_overlap * 4
            temp_chunker = Chunker("fixed", approx_char_size, approx_overlap)
            return temp_chunker._fixed_size_chunking(text)
        
        # Encode entire text
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            start += self.chunk_size - self.chunk_overlap
            
            # Prevent infinite loop
            if self.chunk_size <= self.chunk_overlap:
                break
                
        return chunks

    def _sentence_based_chunking(self, text: str) -> List[str]:
        """
        Sentence-based chunking using NLTK.
        Groups sentences together until reaching target size.
        """
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_size + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Handle overlap: keep last few sentences
                if self.chunk_overlap > 0:
                    overlap_text = ""
                    overlap_sentences = []
                    for sent in reversed(current_chunk):
                        if len(overlap_text) + len(sent) <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_text = " ".join(overlap_sentences)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def _recursive_chunking(self, text):
        """Recursive chunking with improved overlap handling."""
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text, separators):
        """Recursively split text using hierarchical separators."""
        final_chunks = []
        separator = separators[-1]
        
        # Find the best separator
        for sep in separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break
                
        # Split
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
            
        # Merge splits to build chunks with overlap
        current_chunk = ""
        previous_chunk = ""
        
        for s in splits:
            # Calculate potential chunk size
            potential_size = len(current_chunk) + len(s) + len(separator)
            
            if potential_size <= self.chunk_size:
                current_chunk += s + separator
            else:
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())
                    previous_chunk = current_chunk
                
                # Add overlap from previous chunk
                if self.chunk_overlap > 0 and previous_chunk:
                    overlap = previous_chunk[-self.chunk_overlap:]
                    current_chunk = overlap + s + separator
                else:
                    current_chunk = s + separator
        
        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())
            
        return final_chunks

    def _paragraph_chunking(self, text):
        """
        Paragraph-based chunking with overlap support.
        Paragraphs separated by double newlines.
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if self.chunk_overlap == 0:
            return paragraphs
        
        # Add overlap between paragraphs
        chunks = []
        for i, para in enumerate(paragraphs):
            if i == 0:
                chunks.append(para)
            else:
                # Add overlap from previous paragraph
                prev_para = paragraphs[i-1]
                overlap = prev_para[-self.chunk_overlap:] if len(prev_para) > self.chunk_overlap else prev_para
                chunks.append(overlap + "\n\n" + para)
        
        return chunks

def AnyChunkTooBig(chunks, limit):
    """Helper function to check if any chunk exceeds limit."""
    return any(len(c) > limit for c in chunks)

