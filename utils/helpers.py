"""
utils/helpers.py — Shared utilities
"""
import re
import os
import logging
from pathlib import Path
from typing import List


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s → %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(h)
    return logger


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 300) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def get_file_size_mb(filepath: str) -> float:
    return round(os.path.getsize(filepath) / (1024 * 1024), 2)


def format_source_citation(source: str, page=None, chunk_index=None) -> str:
    parts = [f"📄 {source}"]
    if page:
        parts.append(f"Page {page}")
    if chunk_index is not None:
        parts.append(f"Chunk #{chunk_index}")
    return " | ".join(parts)


def validate_file(filepath: str, supported_extensions: List[str],
                  max_size_mb: float = 100.0) -> dict:
    path = Path(filepath)
    if not path.exists():
        return {"valid": False, "reason": f"File not found: {filepath}"}
    ext = path.suffix.lower()
    if ext not in supported_extensions:
        return {"valid": False, "reason": f"Unsupported type '{ext}'. Supported: {supported_extensions}"}
    size = get_file_size_mb(filepath)
    if size > max_size_mb:
        return {"valid": False, "reason": f"File too large: {size} MB (max {max_size_mb} MB)"}
    return {"valid": True, "reason": "OK"}


def deduplicate_chunks(chunks: List[dict]) -> List[dict]:
    seen, unique = set(), []
    for c in chunks:
        t = c.get("text", "").strip()[:100]
        if t not in seen:
            seen.add(t)
            unique.append(c)
    return unique