"""
Document fetching and processing utilities for ArXiv papers.
Reusable across all chunking strategies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import fitz  # pymupdf
import tiktoken


@dataclass
class DocumentBlock:
    """A block of text with hierarchical section info."""

    text: str
    section: Optional[str] = None
    subsection: Optional[str] = None
    block_type: str = "body"  # abstract, section, subsection, body, references


class DocumentProcessor:
    """
    Handles document fetching (PDF extraction) and processing (parsing, tokenization).
    Shared by all chunking strategies.
    """

    def __init__(self):
        self._tokenizer = None

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    def get_tokenizer(self):
        """Get or create tiktoken tokenizer."""
        if self._tokenizer is None:
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        tok = self.get_tokenizer()
        if tok:
            return len(tok.encode(text))
        return len(text) // 4  # Approximate

    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraph blocks."""
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paras) <= 1 and len(text) > 500:
            paras = [p.strip() for p in text.split("\n") if p.strip()]
        return paras

    def get_text_without_abstract(self, text: str) -> str:
        """Return document text with abstract blocks removed."""
        blocks = self.parse_hierarchy(text)
        non_abstract = [b.text for b in blocks if b.block_type != "abstract"]
        return "\n\n".join(non_abstract) if non_abstract else text

    def remove_references(self, text: str) -> str:
        """Remove References/Bibliography section from end of document."""
        ref_end = text.lower().rfind("references")
        if ref_end > len(text) // 2:
            text = text[:ref_end]
        ref_end = text.lower().rfind("bibliography")
        if ref_end > len(text) // 2:
            text = text[:ref_end]
        return text

    def parse_hierarchy(self, text: str) -> List[DocumentBlock]:
        """
        Parse document into hierarchical blocks.
        Detects: Abstract, \\section{}, \\subsection{}, \\subsubsection{}, References.
        """
        blocks: List[DocumentBlock] = []
        lines = text.split("\n")

        section_pattern = re.compile(
            r"^\s*(?:\\section\s*\{([^}]+)\}|(\d+)\s+(.+?))\s*$",
            re.IGNORECASE,
        )
        subsection_pattern = re.compile(
            r"^\s*(?:\\subsection\s*\{([^}]+)\}|(\d+\.\d+)\s+(.+?))\s*$",
            re.IGNORECASE,
        )
        subsubsection_pattern = re.compile(
            r"^\s*(?:\\subsubsection\s*\{([^}]+)\}|(\d+\.\d+\.\d+)\s+(.+?))\s*$",
            re.IGNORECASE,
        )
        refs_pattern = re.compile(
            r"^\s*(?:References?|REFERENCES?|Bibliography|BIBLIOGRAPHY)\s*$",
            re.IGNORECASE,
        )
        abstract_pattern = re.compile(r"^\s*(?:Abstract|ABSTRACT)\s*$", re.IGNORECASE)
        conclusion_pattern = re.compile(
            r"^\s*(?:Conclusion|CONCLUSION|Conclusions?|CONCLUSIONS?)\s*$",
            re.IGNORECASE,
        )

        current_section: Optional[str] = None
        current_subsection: Optional[str] = None
        current_block_type = "body"
        current_text: List[str] = []
        in_references = False

        def flush_block():
            nonlocal current_text, current_block_type
            if current_text:
                block_text = "\n".join(current_text).strip()
                if block_text and not in_references:
                    blocks.append(
                        DocumentBlock(
                            text=block_text,
                            section=current_section,
                            subsection=current_subsection,
                            block_type=current_block_type,
                        )
                    )
            current_text = []
            current_block_type = "body"

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if refs_pattern.match(stripped):
                flush_block()
                in_references = True
                i += 1
                continue

            if abstract_pattern.match(stripped):
                flush_block()
                current_section = "Abstract"
                current_subsection = None
                current_block_type = "abstract"
                current_text = []
                i += 1
                while i < len(lines) and not (
                    section_pattern.match(lines[i].strip())
                    or subsection_pattern.match(lines[i].strip())
                    or refs_pattern.match(lines[i].strip())
                ):
                    current_text.append(lines[i])
                    i += 1
                flush_block()
                current_section = None
                current_subsection = None
                continue

            m = section_pattern.match(stripped)
            if m:
                flush_block()
                name = m.group(1) or m.group(3) or stripped
                current_section = name.strip()
                current_subsection = None
                current_block_type = "conclusion" if conclusion_pattern.match(name) else "section"
                i += 1
                continue

            m = subsection_pattern.match(stripped)
            if m and not subsubsection_pattern.match(stripped):
                flush_block()
                name = m.group(1) or m.group(3) or stripped
                current_subsection = name.strip()
                current_block_type = "subsection"
                i += 1
                continue

            m = subsubsection_pattern.match(stripped)
            if m:
                flush_block()
                name = m.group(1) or m.group(3) or stripped
                current_subsection = name.strip()
                current_block_type = "subsection"
                i += 1
                continue

            if not in_references:
                current_text.append(lines[i])
            i += 1

        flush_block()
        return blocks

    def tokenize(self, text: str) -> tuple:
        """
        Tokenize text. Returns (tokens_list, tokenizer_or_none).
        For char-level fallback when no tokenizer, returns (list of chars, None).
        """
        tok = self.get_tokenizer()
        if tok:
            return tok.encode(text), tok
        return list(text), None
