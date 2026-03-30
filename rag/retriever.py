"""
NUST RAG Retriever
Takes a user query, searches ChromaDB for relevant context,
and returns structured results with confidence scores.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.embedder import ChromaDBManager, db_manager

logger = logging.getLogger("NUSTRetriever")

TOP_N = 5
MIN_CONFIDENCE_THRESHOLD = 55.0  # Enough to filter noise; low enough to retrieve valid chunks

# Patterns that indicate a chunk contains cutoff/merit figures scraped from the web.
# These are blocked so scraped data can never override the authoritative cutoffs.json values.
_CUTOFF_CHUNK_PATTERNS = re.compile(
    r'\b(closing merit|cutoff|cut-off|cut off|last merit|closing aggregate|merit list'
    r'|minimum aggregate|merit.*\d{2,3}[\.,]?\d*\s*%'
    r'|\d{2,3}[\.,]?\d*\s*%.*merit'
    r'|closing.*\d{2,3}[\.,]?\d*\s*%'
    r'|\d{2,3}[\.,]?\d*\s*%.*closing)\b',
    re.I
)

# Prompt-injection patterns — block RAG chunks that look like instructions to the LLM
_INJECTION_CHUNK_PATTERNS = re.compile(
    r'(your\s+task\s*[:is]|write\s+(a|an|me|for|the|it)\s|create\s+(a|an|me)\s|'
    r'generate\s+(a|an)\s|you\s+must\s+|constraints\s*:|document\s+type\s*:|'
    r'without\s+(directly\s+)?quot|new\s+york\s+times|as\s+if\s+(you\s+are|i\s+am|it\s+were)\s|'
    r'instructional\s+guide|plan\s+of\s+action|elaborate\s+narrative|'
    r'developed\s+by\s+(open\s*ai|microsoft)|i\s+am\s+(an?\s+)?ai|as\s+an\s+ai\s+(language\s+)?model|'
    r'provide\s+(me\s+with\s+)?justification|without\s+any\s+formality|'
    r'i\s*\'?s\s+a\s+teacher|dissect\s+(my|the)\s+response|'
    r'write\s+an?\s+(extensive|detailed|comprehensive|personalized)\s|'
    r'from\s+the\s+perspective\s+of\s|tailored\s+specifically\s+towards|'
    r'smithsonian|fbi.style|master.s\s+degree\s+at\s+our\s+university)',
    re.I
)

def _is_cutoff_chunk(text: str) -> bool:
    """Return True if this chunk contains merit cutoff figures that could mislead the LLM."""
    return bool(_CUTOFF_CHUNK_PATTERNS.search(text))

def _is_injection_chunk(text: str) -> bool:
    """Return True if this chunk looks like a prompt-injection attempt."""
    return bool(_INJECTION_CHUNK_PATTERNS.search(text))

# School keywords → canonical school code used in metadata/text
_SCHOOL_QUERY_KEYWORDS: list[tuple[str, str]] = [
    ("seecs", "SEECS"), ("electrical engineering", "SEECS"), ("computer engineering", "SEECS"),
    ("computer science", "SEECS"), ("software engineering", "SEECS"),
    ("information technology", "SEECS"), ("artificial intelligence", "SEECS"),
    ("smme", "SMME"), ("mechanical", "SMME"), ("mechatronics", "SMME"),
    ("scee", "SCEE"), ("civil engineering", "SCEE"), ("environmental engineering", "SCEE"),
    ("scme", "SCME"), ("chemical engineering", "SCME"), ("materials engineering", "SCME"),
    ("nbs", "NBS"), ("business school", "NBS"), ("bba", "NBS"), ("accounting", "NBS"),
    ("sada", "SADA"), ("architecture", "SADA"),
    ("asab", "ASAB"), ("biotechnology", "ASAB"), ("bioinformatics", "ASAB"), ("microbiology", "ASAB"),
    ("sns", "SNS"), ("mathematics", "SNS"), ("physics", "SNS"), ("chemistry", "SNS"),
    ("s3h", "S3H"), ("economics", "S3H"), ("psychology", "S3H"),
    ("igis", "IGIS"), ("gis", "IGIS"),
]


def _detect_school(query: str) -> str | None:
    """Return the school code if a specific school/programme is mentioned in the query."""
    q = query.lower()
    for keyword, code in _SCHOOL_QUERY_KEYWORDS:
        if keyword in q:
            return code
    return None


def _school_relevance_boost(text: str, school: str) -> float:
    """
    Return a boost score (0–10) based on how relevant this chunk is to the target school.
    Chunks that explicitly mention the target school rank higher; chunks that mention
    a DIFFERENT school are penalised so they don't pollute the context.
    """
    text_upper = text.upper()
    all_schools = {"SEECS", "SMME", "SCEE", "SCME", "NBS", "SADA", "ASAB", "SNS", "S3H", "IGIS"}

    if school in text_upper:
        return 10.0  # Strong boost — chunk explicitly mentions target school

    other_schools = all_schools - {school}
    if any(s in text_upper for s in other_schools):
        return -15.0  # Penalise — chunk is about a different school

    return 0.0  # Neutral — general content


def _cosine_distance_to_score(distance: float) -> float:
    """
    Convert ChromaDB cosine distance to a 0-100 confidence score.
    ChromaDB returns distances in [0, 2] for cosine space:
      distance=0 → perfect match → score=100
      distance=1 → orthogonal    → score=50
      distance=2 → opposite      → score=0
    We use: score = (1 - distance/2) * 100
    """
    score = (1.0 - (distance / 2.0)) * 100.0
    return max(0.0, min(100.0, score))


class NUSTRetriever:
    """RAG retriever for NUST knowledge base."""

    def __init__(self, db: ChromaDBManager = None):
        self.db = db or db_manager

    def retrieve(self, query: str) -> dict:
        """
        Retrieve relevant context for the given query.

        Returns:
            {
                "context": str,          # concatenated relevant chunks
                "confidence": float,     # 0-100 avg confidence score
                "sources": [str],        # list of source URLs
                "last_updated": str,     # ISO timestamp of most recent chunk
                "chunks": [dict],        # raw chunk data
            }
        """
        if not query or not query.strip():
            return self._empty_result("Empty query")

        if self.db.get_count() == 0:
            logger.warning("ChromaDB is empty, no context available.")
            return self._empty_result("Knowledge base is empty")

        # Detect school mentioned in query for metadata-aware reranking
        target_school = _detect_school(query)

        # Fetch more candidates than needed so reranking has room to work
        fetch_n = TOP_N * 2 if target_school else TOP_N
        results = self.db.search(query, n=fetch_n)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return self._empty_result("No results found")

        # Score, filter, and rerank candidates
        candidates = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            base_score = _cosine_distance_to_score(dist)

            if base_score < MIN_CONFIDENCE_THRESHOLD:
                continue
            if _is_cutoff_chunk(doc):
                logger.debug("Blocked cutoff chunk: %.60s...", doc)
                continue
            if _is_injection_chunk(doc):
                logger.debug("Blocked injection chunk: %.60s...", doc)
                continue

            # Apply school-relevance boost/penalty when a specific school was detected
            boost = _school_relevance_boost(doc, target_school) if target_school else 0.0
            final_score = base_score + boost

            candidates.append((final_score, base_score, doc, meta))

        # Sort by final score descending, take top N
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:TOP_N]

        context_parts = []
        sources = []
        scores = []
        chunks = []
        timestamps = []

        for final_score, base_score, doc, meta in candidates:
            scores.append(base_score)

            source_url = meta.get("source_url", "nust.edu.pk")
            scraped_at = meta.get("scraped_at", "")
            category   = meta.get("category", "general")

            context_parts.append(f"[{category}]\n{doc}")
            if source_url and source_url not in sources:
                sources.append(source_url)
            if scraped_at:
                timestamps.append(scraped_at)

            chunks.append({
                "text": doc,
                "score": base_score,
                "source_url": source_url,
                "category": category,
                "scraped_at": scraped_at,
            })

        # Average confidence score
        avg_confidence = sum(scores) / len(scores) if scores else 0.0

        # Most recent timestamp
        last_updated = max(timestamps) if timestamps else datetime.now().isoformat()

        context = "\n\n---\n\n".join(context_parts)

        logger.debug(
            "Retrieved %d chunks for query '%s...', avg confidence: %.1f%%",
            len(chunks),
            query[:50],
            avg_confidence,
        )

        return {
            "context": context,
            "confidence": round(avg_confidence, 1),
            "sources": sources,
            "last_updated": last_updated,
            "chunks": chunks,
        }

    def _empty_result(self, reason: str = "") -> dict:
        """Return a fallback empty result."""
        return {
            "context": "",
            "confidence": 0.0,
            "sources": [],
            "last_updated": datetime.now().isoformat(),
            "chunks": [],
            "reason": reason,
        }

    def is_ready(self) -> bool:
        """Return True if the knowledge base has data."""
        return self.db.get_count() > 0


# Singleton
retriever = NUSTRetriever()
