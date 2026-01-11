import requests
import time 

OLLAMA_HOST = "http://localhost:11434"  # running ollama locally
EMBED_MODEL = "mxbai-embed-large"  # an embedding model available in Ollama

def ollama_embed(
    texts,
    max_retries=3,
    sleep_s=2.0,
):
    url = f"{OLLAMA_HOST}/api/embed"
    last_err = None

    for attempt in range(max_retries):
        try:
            r = requests.post(
                url,
                json={"model": EMBED_MODEL, "input": texts},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["embeddings"]

        except requests.RequestException as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(sleep_s * (2 ** attempt))
            else:
                raise RuntimeError(
                    f"Ollama embedding failed after {max_retries} retries"
                ) from e