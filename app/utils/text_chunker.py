"""text_chunker.py – overlapping sliding-window text splitter."""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, document_id: str, extra_metadata: dict | None = None) -> list[TextChunk]:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        if not text:
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks, start = [], 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                end = self._snap(text, start, end) or end
            chunk_text = text[start:end].strip()
            if chunk_text:
                idx = len(chunks)
                chunks.append(TextChunk(
                    chunk_id=f"{document_id}_chunk_{idx}",
                    text=chunk_text,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    metadata={"document_id": document_id, "chunk_index": idx, **(extra_metadata or {})},
                ))
            start += step
        return chunks

    def _snap(self, text: str, start: int, end: int) -> int | None:
        window = text[max(start, end - 50): end]
        for sep in ("\n\n", ". ", "! ", "? "):
            idx = window.rfind(sep)
            if idx != -1:
                return max(start, end - 50) + idx + len(sep)
        return None