# src/rag.py

import os
import json
import pickle
import subprocess
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ============================================================
# PROMPT (V2 – canonical, machine-readable, evaluation-safe)
# ============================================================

PROMPT_V2 = """
You are an expert extraction assistant. ONLY use the CONTEXT below.
Do NOT use any external knowledge.

CONTEXT:
{context}

QUESTION:
{question}

OUTPUT SCHEMA (JSON):
{{
  "answer": "<short, factual answer or empty string>",
  "source_chunks": ["<id1>", "<id2>"],
  "answerable": "yes" | "partial" | "no",
  "notes": "<If partial/no, explain what is missing.>"
}}

RULES:
1. If the full answer is supported by CONTEXT, set "answerable":"yes".
2. If only part is supported, set "answerable":"partial".
3. If nothing answers the question, set "answerable":"no".
4. Use ONLY chunk IDs present in CONTEXT.
5. Be conservative. Do not hallucinate.
""".strip()

# ============================================================
# GLOBAL EMBEDDING MODEL (SINGLE SOURCE OF TRUTH)
# ============================================================

_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts):
    """
    Embed a list of texts into a 2D float32 numpy array.
    This function is used BOTH for indexing and querying.
    """
    embeddings = _EMBED_MODEL.encode(texts, convert_to_numpy=True)

    # Safety: FAISS always expects 2D arrays
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    return embeddings.astype("float32")


# ============================================================
# LOAD FAISS INDEX + METADATA
# ============================================================

def load_index(index_dir):
    index_path = os.path.join(index_dir, "faiss.index")
    ids_path = os.path.join(index_dir, "ids.pkl")
    chunks_path = os.path.join(index_dir, "chunks.json")

    index = faiss.read_index(index_path)

    with open(ids_path, "rb") as f:
        ids = pickle.load(f)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunk_map = json.load(f)

    return index, ids, chunk_map


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(question, index_dir, k=4):
    """
    Retrieve top-k relevant chunks for a question.
    """
    index, ids, chunk_map = load_index(index_dir)

    q_emb = embed_texts([question])
    D, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        cid = ids[idx]
        results.append(chunk_map[cid])

    return results


# ============================================================
# OLLAMA CALL
# ============================================================

def call_ollama(prompt, model="llama3.1:8b"):
    """
    Call local Ollama model via subprocess.
    """
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Force UTF-8 decoding (Windows safe)
    output = result.stdout.decode("utf-8", errors="ignore").strip()
    return output

# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def run_query(question, index_dir):
    """
    End-to-end RAG:
    retrieve → format context → prompt → ollama → answer
    """
    chunks = retrieve(question, index_dir)

    # Format retrieved chunks with IDs (required for grounding)
    context = "\n".join(
        [f"[{c['id']}] {c['text']}" for c in chunks]
    )

    prompt = PROMPT_V2.format(
        context=context,
        question=question
    )

    response = call_ollama(prompt)

    return {
        "question": question,
        "retrieved_chunks": [c["id"] for c in chunks],
        "raw_response": response
    }