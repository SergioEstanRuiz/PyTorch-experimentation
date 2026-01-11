import os
import json
import glob
import time
import numpy as np
import requests
from typing import List, Dict, Any

# either 
OLLAMA_HOST = "http://localhost:11434" # running ollama locally
EMBED_MODEL = "mxbai-embed-large"  # an embedding model available in Ollama
CHUNKS_GLOB = "data/chunks/**/*.json" # location of chunk json files
OUT_DIR = "data/vdb" # output directory for vector db

def ollama_embed(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings from Ollama for a list of texts.
    Returns a list of embeddings (one per input text).
    """
    url = f"{OLLAMA_HOST}/api/embed"
    r = requests.post(url, json={"model": EMBED_MODEL, "input": texts}, timeout=120) # get embeddings for texts
    r.raise_for_status() # raise error if request failed
    data = r.json() # parse json response

    # From the whole response, just get the embeddings part
    embs = data.get("embeddings")
    if embs is None:
        raise RuntimeError(f"Unexpected Ollama response keys: {list(data.keys())}")
    return embs

def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    chunk_paths = sorted(glob.glob(CHUNKS_GLOB, recursive=True)) # returns list of chunk json files
    if not chunk_paths:
        raise SystemExit(f"No chunk json files found for glob: {CHUNKS_GLOB}")

    meta_fpath = os.path.join(OUT_DIR, "meta.jsonl")
    emb_fpath = os.path.join(OUT_DIR, "embeddings.npy")

    meta: List[Dict[str, Any]] = []
    texts: List[str] = []

    for p in chunk_paths:
        # Here we just get the texts from each chunk

        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f) # load the chunk json
        # Expecting obj to contain at least: paper_id, title, text, chunk_id (from prior step)
        meta.append({
            "chunk_id": obj.get("chunk_id"),
            "paper_id": obj.get("paper_id"),
            "title": obj.get("title"),
            "abs_url": obj.get("abs_url"),
            "html_url": obj.get("html_url"),
            "text": obj.get("text"),
        })
        texts.append(obj.get("text", ""))


    all_embs: List[List[float]] = []

    BATCH = 16  # process 16 texts at a time
    # We process in batches to avoid sending too much data at once
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        embs = ollama_embed(batch)
        all_embs.extend(embs)
        time.sleep(0.05)  # tiny pause to be gentle locally

    E = np.array(all_embs, dtype=np.float32)
    E = l2_normalize(E) # normalize embeddings

    # Save meta + embeddings
    with open(meta_fpath, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    np.save(emb_fpath, E)
    print(f"Wrote {len(meta)} chunks")
    print(f"  meta: {meta_fpath}")
    print(f"  embs: {emb_fpath} shape={E.shape}")

if __name__ == "__main__":
    main()
