from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import streamlit as st
from retrieve import top_n 

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

VDB_DIR = APP_DIR / "data" / "vdb"


@contextmanager
def chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


st.title("Mini RAG: Paper Finder")
st.write("Ask about a machine learning topic and get relevant paper chunks.")

with st.form("search"):
    query = st.text_input("ML topic", placeholder="e.g., contrastive learning for vision")
    k = st.slider("Results", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Search")

if submitted:
    if not query.strip():
        st.info("Type a topic to search.")
    elif not VDB_DIR.exists():
        st.error("Vector DB not found. Run src/build_vdb.py to generate data/vdb.")
    else:
        with st.spinner("Retrieving relevant chunks..."):
            try:
                with chdir(VDB_DIR):
                    results = top_n(query, n=k)
            except Exception as exc:
                st.error(f"Search failed: {exc}")
            else:
                if not results:
                    st.warning("No results found.")
                for r in results:
                    title = r.get("title") or "Untitled"
                    score = r.get("score", 0.0)
                    st.subheader(f"{title} ({score:.3f})")
                    abs_url = r.get("abs_url")
                    html_url = r.get("html_url")
                    if abs_url:
                        st.markdown(f"[arXiv]({abs_url})")
                    if html_url:
                        st.markdown(f"[HTML]({html_url})")
                    snippet = (r.get("text") or "").strip()
                    if snippet:
                        preview = snippet[:500] + ("..." if len(snippet) > 500 else "")
                        st.write(preview)
