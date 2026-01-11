import os
import re
import json
import time
import pathlib
import requests
import feedparser
from bs4 import BeautifulSoup

ARXIV_API = "http://export.arxiv.org/api/query" # arxix api to get papers from it instead of scrapping papers

def arxiv_query(search_query: str, max_results: int = 25):
    """
    Query the arXiv API and return the parsed entries.
    Returns a list of feedparser entries.
    """
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    r = requests.get(ARXIV_API, params=params, timeout=30)
    r.raise_for_status()
    return feedparser.parse(r.text).entries

def abs_to_html_url(abs_url: str) -> str:
    # Example abs_url: http://arxiv.org/abs/2402.08954v1
    return abs_url.replace("/abs/", "/html/")

def extract_paragraph_text(html: str) -> str:
    """
    Given a html, you process the paper by getting the paragraphs. 
    Returns the extracted text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Heuristic: grab paragraphs; arXiv HTML usually contains lots of <p>
    ps = soup.find_all("p") # get all paragraph tags
    paras = []
    for p in ps:
        t = p.get_text(" ", strip=True) # get text with spaces
        t = re.sub(r"\s+", " ", t).strip() # get rid of extra spaces/newlines
        if len(t) >= 40:  # only keep reasonably long paragraphs
            paras.append(t)

    return "\n\n".join(paras)

def chunk_words(text: str, chunk_size: int = 250, overlap: int = 50):
    """
    Splits the text into chunks of words with specified size and overlap.
    Returns a list of tuples (start_word_index, end_word_index, chunk_text).
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) >= 200:  # drop very short tail chunks
            chunks.append((start, end, chunk))
        if end == len(words):
            break
    return chunks

def safe_dirname(paper_id: str) -> str:
    # Make a safe directory name from paper_id by replacing unsafe characters.
    return re.sub(r"[^a-zA-Z0-9._-]", "_", paper_id)

def main():
    # “ML papers in general”: use a simple category OR query.
    # You can change this later to keyword queries too.
    query = "(cat:cs.LG OR cat:stat.ML OR cat:cs.CL)" # arXiv query for machine learning papers
    entries = arxiv_query(query, max_results=30) # get latest 30 papers

    out_root = pathlib.Path("data/chunks") # output directory for chunks
    out_root.mkdir(parents=True, exist_ok=True) # create directory if not exists

    session = requests.Session() # create a session for HTTP requests
    session.headers.update({"User-Agent": "mini-rag-demo/0.1 (contact: you@example.com)"}) # set user agent

    for e in entries:
        title = (e.get("title") or "").replace("\n", " ").strip()
        abs_url = e.get("link")
        if not abs_url:
            continue

        paper_id = abs_url.split("/abs/")[-1]  # includes vN
        html_url = abs_to_html_url(abs_url)

        # Check HTML exists (some papers won't have it)
        try:
            resp = session.get(html_url, timeout=30)
            if resp.status_code != 200:
                continue
        except requests.RequestException:
            continue

        full_text = extract_paragraph_text(resp.text)
        if len(full_text) < 2000:  # too little => likely not useful
            continue

        chunks = chunk_words(full_text, chunk_size=250, overlap=50)
        if not chunks:
            continue

        paper_dir = out_root / safe_dirname(paper_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        for i, (w0, w1, ch) in enumerate(chunks):
            obj = {
                "paper_id": paper_id,
                "title": title,
                "abs_url": abs_url,
                "html_url": html_url,
                "chunk_id": f"{paper_id}:{i:04d}",
                "word_start": w0,
                "word_end": w1,
                "text": ch,
            }
            with open(paper_dir / f"{i:04d}.json", "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

        # be polite: avoid hammering arXiv
        time.sleep(1.0)

    print("Done. Wrote chunks under data/chunks/")

if __name__ == "__main__":
    main()
