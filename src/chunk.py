"""
Chunk data model for RAG chunking.
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Chunk:
    """Uniform chunk format for all strategies."""

    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for storage (Qdrant payload)."""
        return {
            "chunk_id": self.id,
            "chunk_text": self.text,
            "paper_id": self.metadata.get("paper_id", ""),
            "title": self.metadata.get("title", ""),
            "section": self.metadata.get("section"),
            "subsection": self.metadata.get("subsection"),
            "position": self.metadata.get("position", 0),
            "strategy": self.metadata.get("strategy", ""),
        }
