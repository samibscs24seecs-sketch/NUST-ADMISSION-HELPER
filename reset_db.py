"""
reset_db.py — Database migration utility.

Clears the ChromaDB collection so the knowledge base can be re-indexed
with new chunking parameters. Run this whenever chunking settings change.

Usage:
    python reset_db.py
"""

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ResetDB")

CHROMA_DIR   = ROOT / "database" / "chroma_store"
HASH_FILE    = ROOT / "data" / "content_hash.json"
KNOWLEDGE_FILE = ROOT / "data" / "nust_knowledge.json"


def clear_chroma() -> bool:
    """Delete the ChromaDB store directory entirely."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        logger.info("Deleted ChromaDB store: %s", CHROMA_DIR)
        return True
    else:
        logger.info("ChromaDB store does not exist — nothing to delete.")
        return False


def clear_hash() -> bool:
    """Delete the content hash file so the scheduler re-scrapes on next run."""
    if HASH_FILE.exists():
        HASH_FILE.unlink()
        logger.info("Deleted content hash: %s", HASH_FILE)
        return True
    return False


def rebuild_from_knowledge() -> int:
    """Re-embed nust_knowledge.json into a fresh ChromaDB collection."""
    if not KNOWLEDGE_FILE.exists():
        logger.error("Knowledge file not found: %s", KNOWLEDGE_FILE)
        return 0

    from rag.embedder import ChromaDBManager
    db = ChromaDBManager()

    with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
        entries = json.load(f)

    now = datetime.now().isoformat()
    chunks = []
    for entry in entries:
        text = entry.get("text", "").strip()
        if text:
            chunks.append({
                "chunk_id":  entry.get("id", f"static_{len(chunks):04d}"),
                "text":      text,
                "source_url": entry.get("source", "https://nust.edu.pk/"),
                "scraped_at": now,
                "category":  entry.get("category", "general"),
            })

    upserted = db.upsert_documents(chunks)
    logger.info("Re-embedded %d documents into ChromaDB.", upserted)
    return upserted


def main():
    print("\n" + "="*60)
    print("  NUST Admission Helper — Database Reset & Re-index")
    print("="*60)

    print("\n[1/3] Clearing ChromaDB store...")
    clear_chroma()

    print("\n[2/3] Clearing content hash (forces fresh scrape on next scheduler run)...")
    clear_hash()

    print("\n[3/3] Re-embedding nust_knowledge.json with new chunk settings...")
    print("      This loads the sentence-transformer model — may take 1-2 minutes.")
    count = rebuild_from_knowledge()

    print("\n" + "="*60)
    if count > 0:
        print(f"  Done. {count} documents embedded.")
        print("  Start the server with: python main.py")
    else:
        print("  Re-embedding failed. Check logs above.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
