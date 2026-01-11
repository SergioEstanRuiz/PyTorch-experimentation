import os
import json
import glob
import time
import numpy as np
import requests
from typing import List, Dict, Any
from src.embedding import ollama_embed

OLLAMA_HOST = "http://localhost:11434" # running ollama locally
EMBED_MODEL = "mxbai-embed-large"  # an embedding model available in Ollama
CHUNKS_GLOB = "data/chunks/**/*.json" # location of chunk json files
OUT_DIR = "data/vdb" # output directory for vector db

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

    batch = 16
    for i in range(0, len(texts), batch):
        print(f"Embedding chunks {i+1}-{min(i+batch, len(texts))}/{len(texts)}", end="\r")
        embs = ollama_embed(texts[i:i+batch])
        all_embs.extend(embs)
        time.sleep(0.1)  # tiny pause to be gentle locally


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
