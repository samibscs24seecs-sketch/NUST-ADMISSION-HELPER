"""
NUST Data Cleaner
Reads raw scraped JSON, cleans text, chunks into ~500 char pieces with 50 char overlap,
and saves cleaned chunks to data/cleaned/
"""

import html
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("NUSTCleaner")

RAW_DIR = ROOT / "data" / "raw"
CLEANED_DIR = ROOT / "data" / "cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE    = 800    # target characters per chunk
CHUNK_OVERLAP = 150    # overlap — enough context to avoid meaning loss at boundaries


def clean_text(text: str) -> str:
    """
    Clean raw extracted text:
    - Decode HTML entities
    - Normalize unicode
    - Collapse whitespace
    - Remove non-printable characters
    """
    if not text:
        return ""

    # Decode HTML entities (e.g., &amp; -> &, &nbsp; -> space)
    text = html.unescape(text)

    # Normalize unicode (NFC form)
    text = unicodedata.normalize("NFC", text)

    # Replace common unicode punctuation with ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2013": "-", "\u2014": "-",   # en/em dash
        "\u2022": "*",                  # bullet
        "\u00a0": " ",                  # non-breaking space
        "\u2026": "...",                # ellipsis
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Remove non-printable / control characters (keep newlines and tabs)
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e\x80-\xff]", " ", text)

    # Collapse multiple spaces and normalize newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    RecursiveCharacterTextSplitter — splits on separators in priority order,
    recursing to the next separator when a piece is still too large.
    Mirrors the LangChain RecursiveCharacterTextSplitter algorithm without
    requiring LangChain as a dependency.

    Separators (in order): double-newline, newline, sentence boundary, space.
    """
    SEPARATORS = ["\n\n", "\n", ". ", " "]

    if not text:
        return []

    def _split(txt: str, separators: list[str]) -> list[str]:
        """Recursively split txt using the first separator that applies."""
        if len(txt) <= chunk_size:
            return [txt] if txt.strip() else []

        sep = separators[0] if separators else " "
        remaining_seps = separators[1:]

        parts = txt.split(sep)

        # Re-attach the separator (except for space — the natural word boundary)
        joiner = sep if sep != " " else " "
        pieces: list[str] = []
        for i, part in enumerate(parts):
            piece = part + (joiner if i < len(parts) - 1 else "")
            pieces.append(piece)

        result: list[str] = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) > chunk_size and remaining_seps:
                result.extend(_split(piece, remaining_seps))
            else:
                result.append(piece)
        return result

    raw_pieces = _split(text, SEPARATORS)

    # Merge small pieces into chunks of up to chunk_size
    chunks: list[str] = []
    current = ""

    for piece in raw_pieces:
        if not piece.strip():
            continue
        candidate = (current + " " + piece).strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece if len(piece) <= chunk_size else piece[:chunk_size]

    if current:
        chunks.append(current)

    if not chunks:
        return []

    # Apply overlap: each chunk starts with the last `overlap` chars of the previous
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        return overlapped

    return chunks


class NUSTCleaner:
    """Cleans raw scraped data and produces structured chunks for embedding."""

    def __init__(self):
        self._chunk_count = 0

    def clean_file(self, raw_file: Path) -> list[dict]:
        """Clean a single raw JSON file and return list of chunk dicts."""
        try:
            with open(raw_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to read %s: %s", raw_file.name, e)
            return []

        source_url = raw.get("url", "")
        category = raw.get("category", "general")
        scraped_at = raw.get("scraped_at", datetime.now().isoformat())

        # Combine main text and tables
        parts = []
        if raw.get("title"):
            parts.append(raw["title"])
        if raw.get("text"):
            parts.append(raw["text"])
        if raw.get("tables"):
            for table in raw["tables"]:
                parts.append(table)

        combined = "\n\n".join(parts)
        cleaned = clean_text(combined)

        if not cleaned:
            logger.warning("No content after cleaning: %s", raw_file.name)
            return []

        text_chunks = chunk_text(cleaned)
        chunks = []

        for idx, chunk_text_content in enumerate(text_chunks):
            if len(chunk_text_content.strip()) < 30:
                continue  # Skip trivially short chunks

            chunk_id = f"{category}_{raw_file.stem}_{idx:04d}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text_content,
                "source_url": source_url,
                "scraped_at": scraped_at,
                "category": category,
            })

        logger.info(
            "Cleaned %s: %d chunks from %d chars",
            raw_file.name, len(chunks), len(cleaned)
        )
        return chunks

    def clean_all(self) -> dict:
        """
        Process all raw JSON files and save cleaned chunks.
        Returns summary dict.
        """
        raw_files = sorted(RAW_DIR.glob("*.json"))
        if not raw_files:
            logger.warning("No raw files found in %s", RAW_DIR)
            return {"chunks_created": 0, "files_processed": 0, "errors": 0}

        all_chunks = []
        errors = 0

        for raw_file in raw_files:
            chunks = self.clean_file(raw_file)
            if chunks:
                all_chunks.extend(chunks)
            else:
                errors += 1

        # Save all chunks to a single cleaned file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = CLEANED_DIR / f"chunks_{timestamp}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        self._chunk_count = len(all_chunks)
        logger.info(
            "Saved %d chunks to %s", len(all_chunks), output_file.name
        )

        return {
            "chunks_created": len(all_chunks),
            "files_processed": len(raw_files),
            "errors": errors,
            "output_file": str(output_file),
        }

    def load_cleaned_chunks(self) -> list[dict]:
        """
        Load the most recent cleaned chunks file.
        Returns list of chunk dicts.
        """
        cleaned_files = sorted(CLEANED_DIR.glob("chunks_*.json"))
        if not cleaned_files:
            logger.warning("No cleaned chunk files found in %s", CLEANED_DIR)
            return []

        latest = cleaned_files[-1]
        try:
            with open(latest, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            logger.info("Loaded %d chunks from %s", len(chunks), latest.name)
            return chunks
        except (json.JSONDecodeError, IOError) as e:
            logger.error("Failed to load %s: %s", latest.name, e)
            return []


# Singleton
cleaner = NUSTCleaner()


if __name__ == "__main__":
    c = NUSTCleaner()
    result = c.clean_all()
    print(json.dumps(result, indent=2))
