import logging
import pickle
import difflib
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 15

# Map job roles to relevant keywords to improve retrieval accuracy
ROLE_EXPANSIONS = {
    "backend": "java spring sql rest microservices linux programming live coding",
    "frontend": "angular javascript react typescript frontend css html web",
    "devops": "aws docker jenkins kubernetes cicd cloud terraform azure",
    "leadership": "management leadership strategy opq opq32r executive",
    "graduate": "entry-level graduate campus scenarios apprentice",
    "sales": "sales transformation revenue account manager business development",
    "admin": "excel word office admin assistant microsoft spreadsheets",
    "customer service": "contact center customer service phone simulation service desk",
    "technical": "programming language linux programming live coding software engineer developer",
    "software": "programming language linux programming live coding developer engineer",
}

STOPWORDS = {
    "a", "an", "the", "for", "in", "on", "of", "to", "is", "are", "with",
    "and", "or", "i", "we", "our", "need", "want", "hire", "hiring", "help",
    "me", "can", "you", "what", "how", "please", "would", "like", "some", "any",
    "test", "assessment"
}

class Retriever:
    """Handles semantic and keyword search over the SHL assessment catalog."""

    def __init__(self, index_path: str, meta_path: str) -> None:
        logger.info(f"Loading FAISS index: {index_path}")
        self.index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            self.metadata: list[dict[str, Any]] = pickle.load(f)

        logger.info(f"Loading transformer model: {EMBED_MODEL}")
        self.model = SentenceTransformer(EMBED_MODEL)

        # Quick lookups for validation and deduplication
        self._name_index = {item["name"].lower().strip(): item for item in self.metadata}
        self._url_index = {item["url"]: item for item in self.metadata}

        logger.info(f"Retriever ready with {len(self.metadata)} items")

    def _embed(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.array(vec, dtype=np.float32)

    def _keyword_score(self, query: str, item: dict[str, Any]) -> float:
        """Token-based overlap score with boost for name matches."""
        query_tokens = {t for t in query.lower().split() if t not in STOPWORDS}
        if not query_tokens:
            return 0.0

        item_name = item.get("name", "").lower()
        content = " ".join([item_name, item.get("description", ""), item.get("test_type", "")]).lower()
        item_tokens = set(content.split())

        overlap = query_tokens & item_tokens
        score = len(overlap) / len(query_tokens)

        # Apply boost if query terms appear in the title
        name_tokens = set(item_name.replace("(", " ").replace(")", " ").split())
        if query_tokens & name_tokens:
            score += 0.25

        return min(1.0, score)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K, filter_levels: list[str] | None = None) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity and keyword overlap."""
        if not query.strip():
            return []

        expanded_query = query.lower()
        for role, keywords in ROLE_EXPANSIONS.items():
            if role in expanded_query:
                expanded_query += f" {keywords}"

        query_vec = self._embed(expanded_query)
        fetch_k = min(top_k * 4, len(self.metadata))
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            item = dict(self.metadata[idx])
            # Hybrid weighting: 70% vector, 30% keyword
            item["_score"] = (0.70 * float(score)) + (0.30 * self._keyword_score(expanded_query, item))
            results.append(item)

        # Sort by hybrid score, using name as tie-breaker
        results.sort(key=lambda x: (x["_score"], x.get("name", "")), reverse=True)

        if filter_levels:
            filtered = [
                r for r in results
                if any(lvl.lower() in [jl.lower() for jl in r.get("job_levels", [])] for lvl in filter_levels)
            ]
            if len(filtered) >= min(3, top_k):
                results = filtered

        return results[:top_k]

    def lookup_by_name(self, name: str) -> dict[str, Any] | None:
        return self._name_index.get(name.lower().strip())

    def fuzzy_lookup(self, name: str, threshold: float = 0.8) -> dict[str, Any] | None:
        """Find catalog entry using fuzzy string matching."""
        name_clean = name.lower().strip()
        candidates = list(self._name_index.keys())
        matches = difflib.get_close_matches(name_clean, candidates, n=1, cutoff=threshold)
        return self._name_index[matches[0]] if matches else None

    def validate_and_resolve(self, names: list[str]) -> list[dict[str, Any]]:
        """Map LLM-generated names back to verified catalog entries."""
        resolved = []
        seen_urls = set()

        for name in names:
            item = self.lookup_by_name(name) or self.fuzzy_lookup(name, threshold=0.75)
            
            if not item:
                logger.debug(f"Dropping unresolvable assessment name: {name}")
                continue
                
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                resolved.append(item)

        return resolved

    @property
    def all_assessments(self) -> list[dict[str, Any]]:
        return self.metadata

_instance = None

def get_retriever() -> Retriever:
    if _instance is None:
        raise RuntimeError("Retriever not initialized. Call init_retriever() first.")
    return _instance

def init_retriever(index_path: str, meta_path: str) -> Retriever:
    global _instance
    _instance = Retriever(index_path, meta_path)
    return _instance
