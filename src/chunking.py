"""
Chunking strategies for academic papers.
Implements 5 specific chunking strategies with metadata support.
"""
import fitz  # pymupdf
import re
import tiktoken
import nltk
from typing import List, Dict, Tuple
from src.section_detector import SectionDetector

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class ChunkMetadata:
    """Metadata for a chunk."""
    def __init__(self, paper_id: str, section_id: str, chunk_id: str, chunk_text: str):
        self.paper_id = paper_id
        self.section_id = section_id
        self.chunk_id = chunk_id
        self.chunk_text = chunk_text
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'paper_id': self.paper_id,
            'section_id': self.section_id,
            'chunk_id': self.chunk_id,
            'chunk_text': self.chunk_text
        }

class Chunker:
    """
    Chunker with 5 specific strategies:
    1. fixed_token: Fixed-length token chunks (~512 tokens, 10-20% overlap)
    2. section: Section-based chunking (one chunk per section)
    3. paragraph: Paragraph-based chunking (merge short paragraphs)
    4. sentence_sliding: Sentence-aware sliding window chunks
    5. section_hybrid: Section + size-capped hybrid
    """
    
    def __init__(self, strategy: str = "fixed_token", chunk_size: int = 512, chunk_overlap: int = 64):
        """
        Initialize chunker with specified strategy.
        
        Args:
            strategy: One of 'fixed_token', 'section', 'paragraph', 'sentence_sliding', 'section_hybrid'
            chunk_size: Target size for chunks (tokens for fixed_token, chars for others)
            chunk_overlap: Overlap size (tokens for fixed_token, chars for others)
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.section_detector = SectionDetector()
        
        # Initialize tokenizer for token-based chunking
        if strategy == "fixed_token":
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            except Exception as e:
                print(f"Warning: Could not load tiktoken encoding: {e}")
                print("Falling back to approximate token counting")
                self.tokenizer = None
        else:
            self.tokenizer = None

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extracts text from PDF bytes, preserving math and LaTeX."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def _get_section_for_position(self, text: str, position: int) -> str:
        """
        Determine which section a given position in text belongs to.
        
        Args:
            text: Full paper text
            position: Character position in text
            
        Returns:
            Section name
        """
        sections = self.section_detector.detect_sections(text)
        
        # Find the section that contains this position
        # Sections are stored as (start_line_index, section_name, section_text)
        # We need to convert line index to character position
        lines = text.split('\n')
        char_pos = 0
        
        for i, line in enumerate(lines):
            if i == position:  # position is actually line index in this context
                # Find which section this line belongs to
                for section_start, section_name, _ in sections:
                    if i >= section_start:
                        # Check if we're still in this section
                        section_end = len(lines)
                        for next_start, _, _ in sections:
                            if next_start > section_start:
                                section_end = next_start
                                break
                        if i < section_end:
                            return section_name
                break
        
        # Fallback: find section by character position
        char_pos = 0
        for i, line in enumerate(lines):
            if char_pos <= position < char_pos + len(line) + 1:  # +1 for newline
                for section_start, section_name, _ in sections:
                    if i >= section_start:
                        section_end = len(lines)
                        for next_start, _, _ in sections:
                            if next_start > section_start:
                                section_end = next_start
                                break
                        if i < section_end:
                            return section_name
                break
            char_pos += len(line) + 1
        
        # If no section found, return first section or "Full Text"
        if sections:
            return sections[0][1]
        return "Full Text"
    
    def _get_section_for_text_snippet(self, text: str, snippet_start: int, snippet_end: int) -> str:
        """
        Determine which section a text snippet belongs to based on its position.
        
        Args:
            text: Full paper text
            snippet_start: Start character position of snippet
            snippet_end: End character position of snippet
            
        Returns:
            Section name
        """
        sections = self.section_detector.detect_sections(text)
        
        if not sections:
            return "Full Text"
        
        # Convert character positions to line positions
        lines = text.split('\n')
        char_pos = 0
        start_line = 0
        end_line = len(lines) - 1
        
        for i, line in enumerate(lines):
            line_end = char_pos + len(line)
            if char_pos <= snippet_start <= line_end:
                start_line = i
            if char_pos <= snippet_end <= line_end:
                end_line = i
                break
            char_pos = line_end + 1  # +1 for newline
        
        # Find section for middle of snippet
        middle_line = (start_line + end_line) // 2
        
        # Find which section contains the middle line
        for i, (section_start, section_name, _) in enumerate(sections):
            # Determine section end
            section_end = len(lines)
            if i + 1 < len(sections):
                section_end = sections[i + 1][0]
            
            if section_start <= middle_line < section_end:
                return section_name
        
        # Fallback: return first section
        return sections[0][1]

    def chunk_paper(self, paper_id: str, text: str) -> List[ChunkMetadata]:
        """
        Chunk a paper with metadata.
        
        Args:
            paper_id: Unique paper identifier
            text: Full paper text
            
        Returns:
            List of ChunkMetadata objects
        """
        if not text:
            return []
        
        if self.strategy == "fixed_token":
            return self._fixed_token_chunking(paper_id, text)
        elif self.strategy == "section":
            return self._section_chunking(paper_id, text)
        elif self.strategy == "paragraph":
            return self._paragraph_chunking(paper_id, text)
        elif self.strategy == "sentence_sliding":
            return self._sentence_sliding_chunking(paper_id, text)
        elif self.strategy == "section_hybrid":
            return self._section_hybrid_chunking(paper_id, text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _fixed_token_chunking(self, paper_id: str, text: str) -> List[ChunkMetadata]:
        """
        Fixed-length token chunks (~512 tokens with 10-20% overlap).
        
        Args:
            paper_id: Paper identifier
            text: Paper text
            
        Returns:
            List of ChunkMetadata objects
        """
        if self.tokenizer is None:
            # Fallback: approximate 1 token ≈ 4 characters
            approx_char_size = self.chunk_size * 4
            approx_overlap = self.chunk_overlap * 4
            chunks = []
            start = 0
            chunk_idx = 0
            
            while start < len(text):
                end = min(start + approx_char_size, len(text))
                chunk_text = text[start:end].strip()
                
                if chunk_text:
                    chunk_id = f"{paper_id}_chunk_{chunk_idx}"
                    # Determine section for this chunk
                    section_id = self._get_section_for_text_snippet(text, start, end)
                    chunks.append(ChunkMetadata(
                        paper_id=paper_id,
                        section_id=section_id,
                        chunk_id=chunk_id,
                        chunk_text=chunk_text
                    ))
                    chunk_idx += 1
                
                start += approx_char_size - approx_overlap
                if self.chunk_size <= self.chunk_overlap:
                    break
            
            return chunks
        
        # Token-based chunking
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        chunk_idx = 0
        
        # Pre-decode full text to get character positions for section detection
        full_text = self.tokenizer.decode(tokens)
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            
            if chunk_text:
                chunk_id = f"{paper_id}_chunk_{chunk_idx}"
                # Estimate character positions for section detection
                # Approximate: find position in decoded text
                decoded_start = len(self.tokenizer.decode(tokens[:start]))
                decoded_end = len(self.tokenizer.decode(tokens[:end]))
                section_id = self._get_section_for_text_snippet(full_text, decoded_start, decoded_end)
                chunks.append(ChunkMetadata(
                    paper_id=paper_id,
                    section_id=section_id,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text
                ))
                chunk_idx += 1
            
            start += self.chunk_size - self.chunk_overlap
            if self.chunk_size <= self.chunk_overlap:
                break
        
        return chunks

    def _section_chunking(self, paper_id: str, text: str) -> List[ChunkMetadata]:
        """
        Section-based chunking (one chunk per logical section).
        
        Args:
            paper_id: Paper identifier
            text: Paper text
            
        Returns:
            List of ChunkMetadata objects
        """
        sections = self.section_detector.detect_sections(text)
        chunks = []
        
        for section_idx, (_, section_name, section_text) in enumerate(sections):
            if section_text.strip():
                chunk_id = f"{paper_id}_section_{section_idx}_{section_name.replace(' ', '_')}"
                chunks.append(ChunkMetadata(
                    paper_id=paper_id,
                    section_id=section_name,
                    chunk_id=chunk_id,
                    chunk_text=section_text
                ))
        
        return chunks

    def _paragraph_chunking(self, paper_id: str, text: str) -> List[ChunkMetadata]:
        """
        Paragraph-based chunking (merge very short paragraphs if needed).
        
        Args:
            paper_id: Paper identifier
            text: Paper text
            
        Returns:
            List of ChunkMetadata objects
        """
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If no double newlines, try single newlines
        if len(paragraphs) <= 1 and len(text) > 1000:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # Merge very short paragraphs (less than 100 chars)
        merged_paragraphs = []
        current_para = ""
        min_para_size = 100
        
        for para in paragraphs:
            if len(current_para) < min_para_size and current_para:
                current_para += "\n\n" + para
            else:
                if current_para:
                    merged_paragraphs.append(current_para)
                current_para = para
        
        if current_para:
            merged_paragraphs.append(current_para)
        
        chunks = []
        char_pos = 0
        for para_idx, para_text in enumerate(merged_paragraphs):
            if para_text.strip():
                chunk_id = f"{paper_id}_para_{para_idx}"
                # Find position of this paragraph in original text
                para_start = text.find(para_text[:min(50, len(para_text))], char_pos)
                if para_start == -1:
                    para_start = char_pos
                para_end = para_start + len(para_text)
                section_id = self._get_section_for_text_snippet(text, para_start, para_end)
                chunks.append(ChunkMetadata(
                    paper_id=paper_id,
                    section_id=section_id,
                    chunk_id=chunk_id,
                    chunk_text=para_text
                ))
                char_pos = para_end
        
        return chunks

    def _sentence_sliding_chunking(self, paper_id: str, text: str) -> List[ChunkMetadata]:
        """
        Sentence-aware sliding window chunks (fixed size, sentence-boundary-aware overlap).
        
        Args:
            paper_id: Paper identifier
            text: Paper text
            
        Returns:
            List of ChunkMetadata objects
        """
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_idx = 0
        char_pos = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds chunk_size, save current chunk
            if current_size + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunk_id = f"{paper_id}_sent_{chunk_idx}"
                    # Find position of chunk in original text
                    chunk_start = text.find(current_chunk[0][:min(50, len(current_chunk[0]))], char_pos)
                    if chunk_start == -1:
                        chunk_start = char_pos
                    chunk_end = chunk_start + len(chunk_text)
                    section_id = self._get_section_for_text_snippet(text, chunk_start, chunk_end)
                    chunks.append(ChunkMetadata(
                        paper_id=paper_id,
                        section_id=section_id,
                        chunk_id=chunk_id,
                        chunk_text=chunk_text
                    ))
                    chunk_idx += 1
                    char_pos = chunk_end
                
                # Handle overlap: keep last few sentences
                if self.chunk_overlap > 0:
                    overlap_sentences = []
                    overlap_size = 0
                    for sent in reversed(current_chunk):
                        if overlap_size + len(sent) <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_size += len(sent) + 1
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunk_id = f"{paper_id}_sent_{chunk_idx}"
                chunk_start = text.find(current_chunk[0][:min(50, len(current_chunk[0]))], char_pos)
                if chunk_start == -1:
                    chunk_start = char_pos
                chunk_end = chunk_start + len(chunk_text)
                section_id = self._get_section_for_text_snippet(text, chunk_start, chunk_end)
                chunks.append(ChunkMetadata(
                    paper_id=paper_id,
                    section_id=section_id,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text
                ))
        
        return chunks

    def _section_hybrid_chunking(self, paper_id: str, text: str) -> List[ChunkMetadata]:
        """
        Section + size-capped hybrid (split by section, then subdivide sections exceeding max token length).
        
        Args:
            paper_id: Paper identifier
            text: Paper text
            
        Returns:
            List of ChunkMetadata objects
        """
        sections = self.section_detector.detect_sections(text)
        chunks = []
        
        # Initialize tokenizer if available
        if self.tokenizer is None:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except:
                self.tokenizer = None
        
        chunk_idx = 0
        
        for section_idx, (_, section_name, section_text) in enumerate(sections):
            if not section_text.strip():
                continue
            
            # Check if section exceeds max size
            if self.tokenizer:
                tokens = self.tokenizer.encode(section_text)
                if len(tokens) <= self.chunk_size:
                    # Section fits in one chunk
                    chunk_id = f"{paper_id}_hybrid_{chunk_idx}_{section_name.replace(' ', '_')}"
                    chunks.append(ChunkMetadata(
                        paper_id=paper_id,
                        section_id=section_name,
                        chunk_id=chunk_id,
                        chunk_text=section_text
                    ))
                    chunk_idx += 1
                else:
                    # Subdivide section using token-based chunking
                    start = 0
                    sub_chunk_idx = 0
                    while start < len(tokens):
                        end = min(start + self.chunk_size, len(tokens))
                        chunk_tokens = tokens[start:end]
                        chunk_text = self.tokenizer.decode(chunk_tokens).strip()
                        
                        if chunk_text:
                            chunk_id = f"{paper_id}_hybrid_{chunk_idx}_{section_name.replace(' ', '_')}_sub{sub_chunk_idx}"
                            chunks.append(ChunkMetadata(
                                paper_id=paper_id,
                                section_id=section_name,
                                chunk_id=chunk_id,
                                chunk_text=chunk_text
                            ))
                            chunk_idx += 1
                            sub_chunk_idx += 1
                        
                        start += self.chunk_size - self.chunk_overlap
                        if self.chunk_size <= self.chunk_overlap:
                            break
            else:
                # Fallback: character-based size check
                if len(section_text) <= self.chunk_size * 4:  # Approximate
                    chunk_id = f"{paper_id}_hybrid_{chunk_idx}_{section_name.replace(' ', '_')}"
                    chunks.append(ChunkMetadata(
                        paper_id=paper_id,
                        section_id=section_name,
                        chunk_id=chunk_id,
                        chunk_text=section_text
                    ))
                    chunk_idx += 1
                else:
                    # Subdivide using character-based chunking
                    start = 0
                    sub_chunk_idx = 0
                    char_size = self.chunk_size * 4
                    char_overlap = self.chunk_overlap * 4
                    
                    while start < len(section_text):
                        end = min(start + char_size, len(section_text))
                        chunk_text = section_text[start:end].strip()
                        
                        if chunk_text:
                            chunk_id = f"{paper_id}_hybrid_{chunk_idx}_{section_name.replace(' ', '_')}_sub{sub_chunk_idx}"
                            chunks.append(ChunkMetadata(
                                paper_id=paper_id,
                                section_id=section_name,
                                chunk_id=chunk_id,
                                chunk_text=chunk_text
                            ))
                            chunk_idx += 1
                            sub_chunk_idx += 1
                        
                        start += char_size - char_overlap
                        if char_size <= char_overlap:
                            break
        
        return chunks
