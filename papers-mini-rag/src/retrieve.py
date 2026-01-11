import os
import json
import numpy as np
import requests
from typing import List, Dict, Any, Tuple

OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"
VDB_DIR = "data/vdb"

def ollama_embed_one(text: str) -> np.ndarray:
    # used to embed query into a normalised vector
    url = f"{OLLAMA_HOST}/api/embed"
    r = requests.post(url, json={"model": EMBED_MODEL, "input": text}, timeout=120)
    r.raise_for_status()
    data = r.json()
    emb = data["embeddings"][0]
    v = np.array(emb, dtype=np.float32)
    v /= max(np.linalg.norm(v), 1e-12)
    return v

def load_vdb() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    # loads the vector database from disk
    meta_path = "meta.jsonl"
    emb_path = "embeddings.npy"

    meta: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    E = np.load(emb_path).astype(np.float32)  # already normalized
    if len(meta) != E.shape[0]:
        raise RuntimeError(f"meta length {len(meta)} != embeddings rows {E.shape[0]}")
    return meta, E

def top_n(query: str, n: int = 5) -> List[Dict[str, Any]]:
    # Returns the top n most relevant chunks for the query
    meta, E = load_vdb()
    q = ollama_embed_one(query)

    scores = E @ q  # cosine since normalized
    if n >= len(scores):
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, n)[:n] # get top n unsorted
        idx = idx[np.argsort(-scores[idx])] # sort top n

    out = []
    for i in idx:
        m = dict(meta[int(i)])
        m["score"] = float(scores[int(i)])
        out.append(m)
    return out
