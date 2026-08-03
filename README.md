# Healthcare Prior Auth Agent

Drafting a prior-authorization (PA) request today means a clinical staffer manually
cross-referencing a member's insurance plan document against a prescription — checking
step therapy, quantity limits, diagnosis eligibility, and prescriber requirements by hand,
then writing up a justification letter. It's slow, repetitive, and error-prone.

This agent automates the first draft: point it at a plan document and a prescription, and
it retrieves the specific plan clauses that apply and drafts a structured PA request letter
that cites them — including calling out any coverage criteria (like step therapy) that
aren't yet documented, instead of silently assuming compliance.

**Who'd use it:** Prior-auth coordinators, payer-facing ops teams at health systems, or
health-tech startups looking to cut the manual drafting time out of PA submissions.

## How it works

1. **Index the plan document** — `POST /plan` accepts either pasted text or a PDF/text
   file upload. The document is chunked (~25-line windows with overlap) and embedded
   locally with `sentence-transformers` (`all-MiniLM-L6-v2` — no paid embeddings API).
2. **Submit the prescription** — `POST /draft` takes drug name, dosage, diagnosis code
   (ICD-10), quantity/frequency, and prescriber/patient fields (synthetic test data only —
   no real PHI is used or required).
3. **Retrieve + draft** — the top-k most relevant plan chunks are retrieved by cosine
   similarity, then handed to the LLM along with a `search_plan_document` tool it can call
   to re-query the plan for anything the initial retrieval didn't surface (e.g. a specific
   quantity limit or exclusion). The model drafts a structured PA letter — Patient/Prescriber
   summary, Medication Requested, Clinical Justification, Plan Coverage Basis (with inline
   `(plan_document:start_line-end_line)` citations), and Requested Determination — and is
   instructed to flag any criteria it can't confirm from the retrieved text rather than
   assume coverage.

## Tech stack

- **Backend:** FastAPI, `pypdf` for PDF text extraction, `sentence-transformers` for local
  embeddings, numpy for similarity search, MiniMax API (OpenAI-compatible) for drafting
  with function calling.
- **Frontend:** React + Vite. Paste/upload plan doc → enter prescription → drafted letter
  with expandable retrieved-clause citations and tool-call trace.
- **LLM:** [MiniMax](https://platform.minimax.io) `MiniMax-M2.7` via its OpenAI-compatible
  endpoint (`https://api.minimax.io/v1`).
- **Docker:** separate images for backend and frontend, plus a `docker-compose.yml`.

## Setup

### Backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set MINIMAX_API_KEY=<your key>
# get a key at https://platform.minimax.io

uvicorn main:app --reload --port 8000
```

The first request will download the `all-MiniLM-L6-v2` embedding model (~90MB), so
indexing a plan document for the first time takes a little longer.

### Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

If the backend isn't on `http://localhost:8000`, copy `frontend/.env.example` to
`frontend/.env` and set `VITE_API_BASE_URL`.

### Docker

```bash
# put your key in backend/.env first (see above)
docker compose up --build
```

## Environment variables

| Var | Where | Default | Purpose |
|---|---|---|---|
| `MINIMAX_API_KEY` | `backend/.env` | *(required)* | MiniMax API key — **never commit this** |
| `MINIMAX_BASE_URL` | `backend/.env` | `https://api.minimax.io/v1` | MiniMax OpenAI-compatible endpoint |
| `MINIMAX_MODEL` | `backend/.env` | `MiniMax-M2.7` | Chat model used for drafting |
| `VITE_API_BASE_URL` | `frontend/.env` | `http://localhost:8000` | Where the frontend finds the backend |

`.env` files are git-ignored; `.env.example` files show the shape without real secrets.

## Demo

A synthetic (non-PHI) plan document and prescription are in `sample_data/`:

- `sample_data/mock_plan_document.txt` — a fictional "Meridian Health Value PPO" pharmacy
  benefit policy with prior-auth criteria for biologic agents (step therapy, quantity
  limits, diagnosis eligibility, prescriber requirements, reauthorization, exclusions).
- `sample_data/mock_prescription.json` — a synthetic Humira (adalimumab) prescription for
  rheumatoid arthritis (ICD-10 M06.9).
- `sample_data/example_draft_response.json` — the actual `/draft` response from running
  that prescription against that plan document end-to-end, including the drafted letter,
  retrieved plan-clause citations, and tool trace.

Running the example: the agent correctly identifies that the diagnosis, quantity, and
prescriber-specialty requirements are met (citing plan Sections 2.1, 2.3, and 2.4), and
flags that step-therapy documentation (Section 2.2) is **not** present in the retrieved
plan context — instead of fabricating compliance, it lists exactly what documentation
would be needed to close that gap.

**Demo GIF/video:** _TODO — add a short recording of indexing a plan document and drafting
a request here._

## Status

Working end-to-end locally, verified against the synthetic plan/prescription in
`sample_data/` (see above). Not yet deployed anywhere — no live demo link. Deploying (e.g.
to Fly.io / Cloud Run + Vercel) is a manual follow-up.

### Known rough edges

- In-memory session store means indexed plan documents don't survive a backend restart,
  and memory grows with each indexed document (no eviction). Fine for a demo, not for
  production multi-tenant use.
- Chunking is fixed-size by line count, not section-aware — it doesn't align chunk
  boundaries to plan document headings/clauses.
- Only tested against a single synthetic plan document layout; real-world plan documents
  vary widely in structure and may need chunking/prompt tuning.
- The frontend Docker image bakes `VITE_API_BASE_URL` in at build time (standard Vite
  behavior for static builds); pointing a built frontend image at a different backend URL
  requires rebuilding it.
