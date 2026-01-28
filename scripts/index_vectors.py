# scripts/index_vectors.py

import os
import json
import pickle
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# GLOBAL EMBEDDING MODEL (MUST MATCH src/rag.py)
# ============================================================

_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    """
    Embed a list of texts into a 2D float32 numpy array.
    Handles single or multiple texts safely.
    """
    emb = _EMBED_MODEL.encode(texts, convert_to_numpy=True)

    # Safety: FAISS always expects 2D
    if emb.ndim == 1:
        emb = np.expand_dims(emb, axis=0)

    return emb.astype("float32")


# ============================================================
# MAIN INDEXING LOGIC
# ============================================================

def main(chunks_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load chunks
    # --------------------------------------------------------
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    if len(chunks) == 0:
        raise ValueError("No chunks found. Cannot build FAISS index.")

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    # --------------------------------------------------------
    # Embed chunks (SAME CODE PATH AS QUERY)
    # --------------------------------------------------------
    embeddings = embed_texts(texts)

    # --------------------------------------------------------
    # Build FAISS index
    # --------------------------------------------------------
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # --------------------------------------------------------
    # Persist index and metadata
    # --------------------------------------------------------
    faiss.write_index(index, os.path.join(out_dir, "faiss.index"))

    with open(os.path.join(out_dir, "ids.pkl"), "wb") as f:
        pickle.dump(ids, f)

    chunk_map = {c["id"]: c for c in chunks}
    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)

    print("FAISS index created")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--out", required=True, help="Output directory for FAISS index")
    args = parser.parse_args()

    main(args.chunks, args.out)