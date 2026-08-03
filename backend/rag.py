"""Plan document ingestion, chunking, embedding, and retrieval."""
import io
import uuid

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_LINES = 25
CHUNK_OVERLAP = 6

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def extract_text(filename: str, file_bytes: bytes) -> str:
    if filename and filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    return file_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, source_name: str = "plan_document"):
    lines = text.splitlines()
    chunks = []
    start = 0
    n = len(lines)
    if n == 0:
        return chunks
    while start < n:
        end = min(start + CHUNK_LINES, n)
        chunk_lines = lines[start:end]
        body = "\n".join(chunk_lines).strip()
        if body:
            chunks.append({
                "source": source_name,
                "start_line": start + 1,
                "end_line": end,
                "text": body,
            })
        if end == n:
            break
        start = end - CHUNK_OVERLAP
    return chunks


def build_index(text: str, source_name: str = "plan_document") -> dict:
    chunks = chunk_text(text, source_name)
    if not chunks:
        raise RuntimeError("No extractable text found in the plan document")

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

    session_id = str(uuid.uuid4())
    return {
        "session_id": session_id,
        "source_name": source_name,
        "raw_text": text,
        "chunks": chunks,
        "embeddings": np.asarray(embeddings, dtype=np.float32),
        "num_chunks": len(chunks),
    }


def retrieve(session: dict, query: str, top_k: int = 6):
    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    sims = session["embeddings"] @ q_emb
    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for idx in top_idx:
        chunk = session["chunks"][idx]
        results.append({**chunk, "score": float(sims[idx])})
    return results
