"""Plan document ingestion, chunking, embedding, and retrieval."""
import io
import re
import uuid

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

CHUNK_LINES = 25
CHUNK_OVERLAP = 6

# Plan documents are hierarchical (SECTION -> numbered clause -> qualifying
# sub-bullet). Flat fixed-line chunking can split a rule from the condition
# that qualifies it (e.g. "2 pens per 28 days" ends up in a different chunk
# than "...unless dose escalation is documented"). We chunk on structural
# boundaries first and only fall back to a line-window split for sections
# that are themselves too long to embed as one chunk.
_HEADING_RE = re.compile(
    r"^\s*("
    r"SECTION\s+\d+[:.]?.*"          # SECTION 2: BIOLOGIC AGENTS ...
    r"|\d+\.\d+\s.+"                 # 2.2 Step therapy requirement: ...
    r"|[A-Z][A-Z \-/]{6,}$"          # ALL-CAPS HEADING LINES
    r")\s*$"
)
MAX_SECTION_LINES = 40  # sections longer than this still get sub-chunked

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


def _split_into_sections(lines: list) -> list:
    """Group lines into (start_idx, end_idx) spans, breaking at heading lines.
    A heading line starts a new section that includes everything up to (but
    not including) the next heading. Leading non-heading lines (a preamble
    before the first heading) form their own section."""
    boundaries = [i for i, line in enumerate(lines) if _HEADING_RE.match(line)]
    if not boundaries:
        return [(0, len(lines))]

    spans = []
    if boundaries[0] > 0:
        spans.append((0, boundaries[0]))
    for i, b in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        spans.append((b, end))
    return spans


def chunk_text(text: str, source_name: str = "plan_document"):
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []

    chunks = []
    for span_start, span_end in _split_into_sections(lines):
        span_len = span_end - span_start
        if span_len <= MAX_SECTION_LINES:
            body = "\n".join(lines[span_start:span_end]).strip()
            if body:
                chunks.append({
                    "source": source_name,
                    "start_line": span_start + 1,
                    "end_line": span_end,
                    "text": body,
                })
            continue

        # Section too long to embed as one chunk (e.g. a wall-of-text
        # clause list) — fall back to overlapping line windows within it,
        # so we never lose structural grouping *and* never produce an
        # oversized, poorly-embedded chunk.
        start = span_start
        while start < span_end:
            end = min(start + CHUNK_LINES, span_end)
            body = "\n".join(lines[start:end]).strip()
            if body:
                chunks.append({
                    "source": source_name,
                    "start_line": start + 1,
                    "end_line": end,
                    "text": body,
                })
            if end == span_end:
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


KEYWORD_BOOST = 0.05


def retrieve(session: dict, query: str, top_k: int = 6, boost_terms: list = None):
    """Cosine-similarity retrieval, optionally with a small keyword-match boost.

    Pure embedding similarity can surface a plausible-but-wrong clause from an
    unrelated section (e.g. a quantity-limit clause for a different drug
    class). `boost_terms` is a cheap, explicit routing signal — drug name /
    diagnosis code terms the caller already knows are relevant — nudging
    chunks that literally mention them above chunks that are only
    semantically similar. This isn't a substitute for real hierarchical
    retrieval (see README), but it materially reduces cross-section mixups
    for something this local and free.
    """
    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    # On some BLAS backends (observed with Apple Accelerate) this matmul raises
    # spurious divide-by-zero/overflow/invalid RuntimeWarnings even though both
    # operands are finite unit vectors and the result is itself finite — a
    # known FP-flag false positive, not an actual NaN/Inf propagation. We
    # verified this by inspecting `sims` directly; still, guard defensively
    # in case a genuinely degenerate (all-zero) embedding ever produces a
    # real NaN, so it sorts last instead of silently ranking first.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sims = session["embeddings"] @ q_emb
    sims = np.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)

    if boost_terms:
        terms = [t.lower() for t in boost_terms if t and len(t) > 2]
        for i, chunk in enumerate(session["chunks"]):
            text_lower = chunk["text"].lower()
            if any(term in text_lower for term in terms):
                sims[i] += KEYWORD_BOOST

    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for idx in top_idx:
        chunk = session["chunks"][idx]
        results.append({**chunk, "score": float(sims[idx])})
    return results
