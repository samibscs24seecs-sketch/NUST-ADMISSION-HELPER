"""
ChromaDB Manager for NUST RAG system.
Uses sentence-transformers all-MiniLM-L6-v2 for local embeddings.
"""

import logging
import re
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("ChromaDBManager")

# Block injection content at write time — defence-in-depth alongside the retriever filter.
# Scraped pages sometimes contain prompt-injection or unrelated content;
# reject them before they ever enter the vector store.
_WRITE_INJECTION_PATTERN = re.compile(
    r'(your\s+task\s*[:is]|write\s+(a|an|me|for|the|it)\s|create\s+(a|an|me)\s|'
    r'generate\s+(a|an)\s|you\s+must\s+|new\s+york\s+times|smithsonian|'
    r'as\s+if\s+(you\s+are|i\s+am|it\s+were)\s|instructional\s+guide|'
    r'plan\s+of\s+action|elaborate\s+narrative|developed\s+by\s+(open\s*ai|microsoft)|'
    r'i\s+am\s+(an?\s+)?ai\b|as\s+an\s+ai\s+(language\s+)?model|'
    r'write\s+an?\s+(extensive|elaborate|detailed|comprehensive)\s|'
    r'from\s+the\s+perspective\s+of\s|tailored\s+specifically\s+towards|'
    r'fbi.style|death\s+benefit|baby\s+boom|documentary\s+hypothesis|'
    r'cancellation\s+patterns|essay\s+on\s+an?|craft\s+an?\s+(elaborate|detailed))',
    re.I
)


def _is_safe_chunk(text: str) -> bool:
    """Return True if this chunk is safe to store in ChromaDB."""
    return not bool(_WRITE_INJECTION_PATTERN.search(text))


CHROMA_DIR = ROOT / "database" / "chroma_store"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "nust_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ChromaDBManager:
    """
    Manages ChromaDB vector store with sentence-transformer embeddings.
    Runs 100% locally — no API keys required.
    """

    def __init__(self):
        self._client = None
        self._collection = None
        self._embedding_fn = None
        self._initialized = False
        self._init_error: Optional[str] = None

    def _initialize(self):
        """Lazy initialization — import heavy dependencies on first use."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions

            # Persistent ChromaDB client
            self._client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )

            # Sentence-transformer embedding function
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            self._initialized = True
            logger.info(
                "ChromaDB initialized. Collection '%s' has %d documents.",
                COLLECTION_NAME,
                self._collection.count(),
            )

        except ImportError as e:
            self._init_error = f"Missing dependency: {e}"
            logger.error("ChromaDB/sentence-transformers not installed: %s", e)
        except Exception as e:
            self._init_error = str(e)
            logger.error("ChromaDB initialization failed: %s", e)

    def is_ready(self) -> bool:
        """Return True if ChromaDB is initialized and ready."""
        if not self._initialized:
            self._initialize()
        return self._initialized and self._collection is not None

    def add_documents(self, chunks: list[dict]) -> int:
        """
        Add document chunks to ChromaDB.
        Each chunk: {chunk_id, text, source_url, scraped_at, category}
        Returns number of documents added.
        """
        if not self.is_ready():
            logger.error("ChromaDB not ready, cannot add documents.")
            return 0

        if not chunks:
            return 0

        # Deduplicate by chunk_id (upsert logic: delete then add)
        existing_ids = set()
        try:
            results = self._collection.get(include=[])
            existing_ids = set(results.get("ids", []))
        except Exception:
            pass

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "").strip()

            if not text or not chunk_id:
                continue

            # Skip if already exists
            if chunk_id in existing_ids:
                continue

            if not _is_safe_chunk(text):
                logger.warning("Blocked injection chunk at write time: %.80s...", text)
                continue

            documents.append(text)
            metadatas.append({
                "source_url": chunk.get("source_url", ""),
                "scraped_at": chunk.get("scraped_at", ""),
                "category": chunk.get("category", "general"),
                "chunk_id": chunk_id,
            })
            ids.append(chunk_id)

        if not ids:
            logger.info("No new documents to add (all already exist).")
            return 0

        # Batch add in groups of 100
        batch_size = 100
        total_added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            try:
                self._collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids,
                )
                total_added += len(batch_ids)
                logger.info("Added batch %d-%d (%d docs)", i, i + len(batch_ids), len(batch_ids))
            except Exception as e:
                logger.error("Error adding batch %d: %s", i, e)

        logger.info("Total documents added: %d", total_added)
        return total_added

    def upsert_documents(self, chunks: list[dict]) -> int:
        """
        Upsert (add or update) document chunks in ChromaDB.
        Returns number of documents upserted.
        """
        if not self.is_ready():
            logger.error("ChromaDB not ready.")
            return 0

        if not chunks:
            return 0

        documents = []
        metadatas = []
        ids = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "").strip()
            if not text or not chunk_id:
                continue
            if not _is_safe_chunk(text):
                logger.warning("Blocked injection chunk at write time: %.80s...", text)
                continue
            documents.append(text)
            metadatas.append({
                "source_url": chunk.get("source_url", ""),
                "scraped_at": chunk.get("scraped_at", ""),
                "category": chunk.get("category", "general"),
                "chunk_id": chunk_id,
            })
            ids.append(chunk_id)

        if not ids:
            return 0

        batch_size = 100
        total = 0
        for i in range(0, len(ids), batch_size):
            try:
                self._collection.upsert(
                    documents=documents[i : i + batch_size],
                    metadatas=metadatas[i : i + batch_size],
                    ids=ids[i : i + batch_size],
                )
                total += len(ids[i : i + batch_size])
            except Exception as e:
                logger.error("Upsert batch error at %d: %s", i, e)

        return total

    def search(self, query: str, n: int = 3) -> dict:
        """
        Search ChromaDB for documents similar to query.
        Returns dict with documents, metadatas, distances.
        """
        if not self.is_ready():
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        if self._collection.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(n, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            logger.error("Search error: %s", e)
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def clear_collection(self) -> bool:
        """Delete and recreate the collection (clears all data)."""
        if not self.is_ready():
            return False

        try:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Collection '%s' cleared.", COLLECTION_NAME)
            return True
        except Exception as e:
            logger.error("Failed to clear collection: %s", e)
            return False

    def get_count(self) -> int:
        """Return number of documents in the collection."""
        if not self.is_ready():
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    @staticmethod
    def quick_count() -> int:
        """
        Return document count WITHOUT loading the embedding model.
        Safe to call at startup — only opens the ChromaDB client, no sentence-transformers.
        Returns 0 if the collection does not exist yet.
        """
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=Settings(anonymized_telemetry=False),
            )
            col = client.get_collection(name=COLLECTION_NAME)
            return col.count()
        except Exception:
            return 0


# Singleton
db_manager = ChromaDBManager()
