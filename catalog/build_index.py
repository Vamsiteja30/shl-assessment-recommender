import json
import logging
import pickle
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CATALOG_PATH = Path("data/shl_product_catalog.json")
INDEX_PATH = Path("data/faiss_index.bin")
META_PATH = Path("data/faiss_meta.pkl")
EMBED_MODEL = "all-MiniLM-L6-v2"

# Priority for selecting a single representative category for an assessment
KEY_PRIORITY = [
    "Ability & Aptitude",
    "Biodata & Situational Judgment",
    "Competencies",
    "Development & 360",
    "Assessment Exercises",
    "Knowledge & Skills",
    "Personality & Behavior",
    "Simulations",
]

def get_primary_test_type(keys: list[str]) -> str:
    if not keys:
        return "Knowledge & Skills"
    for k in KEY_PRIORITY:
        if k in keys:
            return k
    return keys[0]

def build_document(item: dict[str, Any]) -> str:
    """Concatenate catalog fields into a single searchable text string."""
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        f"Types: {', '.join(item.get('keys', []))}",
        f"Levels: {', '.join(item.get('job_levels', []))}",
        f"Duration: {item.get('duration', '')}",
        f"Languages: {', '.join(item.get('languages', [])[:3])}"
    ]
    
    if item.get("remote") == "yes":
        parts.append("Remote testing available")
    if item.get("adaptive") == "yes":
        parts.append("Adaptive test")
        
    return ". ".join(filter(None, parts))

def load_catalog(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f, strict=False)
    
    valid = [
        item for item in data
        if item.get("status") in ("ok", None, "")
        and item.get("link", "").startswith("http")
        and item.get("name", "").strip()
    ]
    logger.info(f"Loaded {len(valid)} valid assessments from {path}")
    return valid

def build_index(catalog_path: Path, index_path: Path, meta_path: Path) -> None:
    catalog = load_catalog(catalog_path)
    documents = [build_document(item) for item in catalog]

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    logger.info(f"Embedding {len(documents)} items...")
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # Build FAISS index for similarity search
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built with {index.ntotal} vectors")

    # Extract relevant metadata for lookup
    meta = []
    for item in catalog:
        meta.append({
            "name": item["name"],
            "url": item["link"],
            "test_type": get_primary_test_type(item.get("keys", [])),
            "keys": item.get("keys", []),
            "description": item.get("description", ""),
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", []),
            "duration": item.get("duration", ""),
            "remote": item.get("remote", ""),
            "adaptive": item.get("adaptive", ""),
        })

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    logger.info(f"Index and metadata saved to {index_path} and {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(CATALOG_PATH))
    parser.add_argument("--index", default=str(INDEX_PATH))
    parser.add_argument("--meta", default=str(META_PATH))
    args = parser.parse_args()

    build_index(Path(args.catalog), Path(args.index), Path(args.meta))
