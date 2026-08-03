import os

from dotenv import load_dotenv

load_dotenv()

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag
from minimax_client import draft_prior_auth

app = FastAPI(title="Healthcare Prior Auth Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> {chunks, embeddings, raw_text, ...}
SESSIONS: dict = {}


class PrescriptionRequest(BaseModel):
    session_id: str
    drug_name: str
    dosage: str
    diagnosis_code: str
    quantity: Optional[str] = None
    prescriber_name: Optional[str] = None
    patient_name: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/plan")
async def index_plan(
    plan_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not plan_text and not file:
        raise HTTPException(400, "Provide either plan_text or a file upload")

    if file:
        file_bytes = await file.read()
        try:
            text = rag.extract_text(file.filename, file_bytes)
        except Exception as e:
            raise HTTPException(400, f"Failed to extract text from file: {e}")
        source_name = file.filename or "plan_document"
    else:
        text = plan_text
        source_name = "plan_document"

    try:
        session = rag.build_index(text, source_name)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    SESSIONS[session["session_id"]] = session
    return {
        "session_id": session["session_id"],
        "source_name": session["source_name"],
        "num_chunks": session["num_chunks"],
    }


@app.post("/draft")
def draft(req: PrescriptionRequest):
    session = SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, "Unknown session_id — index a plan document first via /plan")

    if not os.environ.get("MINIMAX_API_KEY"):
        raise HTTPException(500, "MINIMAX_API_KEY is not configured on the server")

    query = f"{req.drug_name} {req.dosage} {req.diagnosis_code} prior authorization coverage criteria"
    retrieved = rag.retrieve(session, query, top_k=6)

    prescription = req.model_dump()
    result = draft_prior_auth(prescription, retrieved, session)

    return {
        "draft": result["draft"],
        "citations": [
            {"source": c["source"], "start_line": c["start_line"], "end_line": c["end_line"], "score": c["score"]}
            for c in retrieved
        ],
        "tool_trace": result["tool_trace"],
    }


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Unknown session_id")
    del SESSIONS[session_id]
    return {"status": "deleted"}
