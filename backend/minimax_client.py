"""Thin wrapper around MiniMax's OpenAI-compatible chat completions API,
with a manual tool-calling loop for re-querying the indexed plan document."""
import json
import os
import re

from openai import OpenAI

from tools import TOOL_SCHEMAS, execute_tool_call

# MiniMax M2.x models always emit a <think>...</think> block and it cannot be
# disabled (per MiniMax docs); strip it before returning the draft to users.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


MINIMAX_BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """You are a prior-authorization drafting agent for a health plan's \
pharmacy benefit. You are given a prescription (drug, dosage, diagnosis) and retrieved \
clauses from the patient's insurance plan document, plus a tool to search that plan \
document further when the retrieved context doesn't cover something you need to check \
(e.g. step therapy requirements, quantity limits, age restrictions, or diagnosis-specific \
coverage criteria).

Rules:
- Ground every coverage claim in the actual plan text you were shown or fetched via tools.
  Never invent a plan rule that isn't in the retrieved text.
- Cite the plan document inline using the format (plan_document:start_line-end_line) next \
  to every clause you reference.
- Produce a formal, structured prior-authorization request letter with these sections: \
  Patient/Prescriber summary (use the fields given, do not invent identifying details), \
  Medication Requested, Clinical Justification, Plan Coverage Basis (with citations), and \
  Requested Determination.
- If the plan document doesn't address something needed (e.g. no explicit step-therapy \
  clause found), state that plainly rather than guessing, and note it as a gap for the \
  preparer to confirm manually.
- Be precise and professional; this letter may be submitted to a payer.
"""


def _client() -> OpenAI:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set (check your .env file)")
    return OpenAI(api_key=api_key, base_url=MINIMAX_BASE_URL)


def _build_context_block(chunks: list) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"### {c['source']}:{c['start_line']}-{c['end_line']} (score={c['score']:.3f})\n"
            f"```\n{c['text']}\n```"
        )
    return "\n\n".join(parts)


def draft_prior_auth(prescription: dict, retrieved_chunks: list, session: dict) -> dict:
    client = _client()

    rx_summary = (
        f"Drug: {prescription['drug_name']}\n"
        f"Dosage/Strength: {prescription['dosage']}\n"
        f"Diagnosis code (ICD-10): {prescription['diagnosis_code']}\n"
        f"Quantity/Frequency: {prescription.get('quantity', 'not specified')}\n"
        f"Prescriber: {prescription.get('prescriber_name', 'not specified')}\n"
        f"Patient (synthetic/test): {prescription.get('patient_name', 'not specified')}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Prescription details:\n{rx_summary}\n\n"
                f"Retrieved plan document context:\n{_build_context_block(retrieved_chunks)}\n\n"
                "Use the search_plan_document tool if you need to check something not "
                "covered by the retrieved context above (e.g. step therapy, quantity "
                "limits, exclusions). Otherwise draft the prior-authorization request now."
            ),
        },
    ]

    tool_trace = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MINIMAX_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )
        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            return {
                "draft": _strip_thinking(msg.content),
                "tool_trace": tool_trace,
            }

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool_call(session, tc.function.name, args)
            tool_trace.append({"tool": tc.function.name, "args": args, "result": result[:2000]})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Ran out of tool rounds; force a final draft without tools.
    final = client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=messages + [{"role": "user", "content": "Please produce the final draft now, citing sources."}],
        temperature=0.2,
    )
    return {
        "draft": _strip_thinking(final.choices[0].message.content),
        "tool_trace": tool_trace,
    }
