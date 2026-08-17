"""Structured criteria extraction + deterministic checklist evaluation.

This is the "rules engine decides, LLM only narrates" split applied to plan
documents. Two stages, deliberately separated:

1. extract_criteria() — a single, *constrained* LLM call (tool_choice forces
   one function call, not free text) that pulls structured fields out of the
   retrieved plan chunks: covered ICD-10 prefixes, step-therapy requirement,
   quantity limit, required prescriber specialties, exclusions — each with a
   citation back to the source chunk. The LLM is only asked to *transcribe*
   what's in the text into a schema, not to judge whether a given
   prescription satisfies it.

2. check_criteria() — plain Python, no LLM. Compares the prescription against
   the extracted criteria and produces a checklist where every item is one of
   "met" / "not_met" / "needs_review". "needs_review" is the load-bearing
   status: whenever the deterministic check can't confidently resolve a
   criterion (missing data, fuzzy quantity units, ambiguous specialty match),
   it routes to manual review instead of guessing — the same bucket PBM
   Certis routes an unclassifiable mismatch to, applied here to coverage
   criteria instead of claims fields.

The drafting LLM call (see minimax_client.draft_prior_auth) is handed this
checklist as ground truth and instructed to narrate it, not re-derive it —
so the parts of the decision that *are* structured/checkable never depend on
the LLM getting a judgment call right from raw prose.
"""
import json
import os
import re

from openai import OpenAI

MINIMAX_BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
MINIMAX_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")

EXTRACTION_SYSTEM_PROMPT = """You extract structured coverage criteria from health \
insurance plan document excerpts. You do not judge whether any particular patient or \
prescription satisfies the criteria — you only transcribe what the plan text says into \
the given schema. If a field isn't addressed in the provided text, leave it null/empty \
rather than inferring or guessing. Every non-null field must include a citation in the \
form "source:start_line-end_line" copied from the chunk heading it came from."""

_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "record_plan_criteria",
        "description": "Record the structured prior-authorization criteria found in the plan document excerpts.",
        "parameters": {
            "type": "object",
            "properties": {
                "covered_icd10_prefixes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "ICD-10 code prefixes explicitly listed as covered diagnoses, e.g. ['M05', 'M06', 'L40.5']. Strip trailing '.-' or '-'.",
                },
                "covered_icd10_citation": {"type": "string", "description": "source:start_line-end_line, or empty if not found"},
                "step_therapy_required": {
                    "type": ["boolean", "null"],
                    "description": "true if the text requires trying another drug first, false if explicitly not required, null if not addressed",
                },
                "step_therapy_detail": {"type": "string", "description": "The step therapy rule text, verbatim or closely paraphrased. Empty if not addressed."},
                "step_therapy_citation": {"type": "string", "description": "source:start_line-end_line, or empty if not found"},
                "quantity_limit_text": {"type": "string", "description": "The quantity limit as stated, e.g. '2 pens (40mg each) per 28-day supply'. Empty if not addressed."},
                "quantity_limit_citation": {"type": "string", "description": "source:start_line-end_line, or empty if not found"},
                "prescriber_specialty_required": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specialist types explicitly required to submit/co-sign the request, e.g. ['rheumatologist', 'gastroenterologist']. Empty list if not addressed.",
                },
                "prescriber_specialty_citation": {"type": "string", "description": "source:start_line-end_line, or empty if not found"},
                "prior_auth_required": {
                    "type": ["boolean", "null"],
                    "description": "true if the excerpts state this drug class requires prior authorization, null if not addressed",
                },
                "exclusions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit exclusions mentioned (e.g. 'cosmetic use', 'off-label use'). Empty list if none found.",
                },
            },
            "required": ["covered_icd10_prefixes", "step_therapy_required", "quantity_limit_text", "prescriber_specialty_required", "prior_auth_required", "exclusions"],
        },
    },
}


def _client() -> OpenAI:
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set (check your .env file)")
    return OpenAI(api_key=api_key, base_url=MINIMAX_BASE_URL)


def _build_context_block(chunks: list) -> str:
    parts = []
    for c in chunks:
        parts.append(f"### {c['source']}:{c['start_line']}-{c['end_line']}\n```\n{c['text']}\n```")
    return "\n\n".join(parts)


def extract_criteria(retrieved_chunks: list) -> dict:
    """Single constrained LLM call: transcribe plan text into a criteria schema."""
    client = _client()
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Plan document excerpts:\n\n"
                f"{_build_context_block(retrieved_chunks)}\n\n"
                "Call record_plan_criteria with what these excerpts say. Leave fields "
                "empty/null if not addressed here — do not guess."
            ),
        },
    ]
    response = client.chat.completions.create(
        model=MINIMAX_MODEL,
        messages=messages,
        tools=[_EXTRACT_TOOL],
        tool_choice={"type": "function", "function": {"name": "record_plan_criteria"}},
        temperature=0,
    )
    msg = response.choices[0].message
    if not msg.tool_calls:
        # Model declined to call the tool (e.g. nothing extractable) — return an
        # all-empty schema rather than raising, so the checklist below can
        # route every item to needs_review instead of the endpoint erroring.
        return _empty_criteria()
    try:
        args = json.loads(msg.tool_calls[0].function.arguments or "{}")
    except json.JSONDecodeError:
        return _empty_criteria()

    return {
        "covered_icd10_prefixes": args.get("covered_icd10_prefixes") or [],
        "covered_icd10_citation": args.get("covered_icd10_citation") or "",
        "step_therapy_required": args.get("step_therapy_required"),
        "step_therapy_detail": args.get("step_therapy_detail") or "",
        "step_therapy_citation": args.get("step_therapy_citation") or "",
        "quantity_limit_text": args.get("quantity_limit_text") or "",
        "quantity_limit_citation": args.get("quantity_limit_citation") or "",
        "prescriber_specialty_required": args.get("prescriber_specialty_required") or [],
        "prescriber_specialty_citation": args.get("prescriber_specialty_citation") or "",
        "prior_auth_required": args.get("prior_auth_required"),
        "exclusions": args.get("exclusions") or [],
    }


def _empty_criteria() -> dict:
    return {
        "covered_icd10_prefixes": [], "covered_icd10_citation": "",
        "step_therapy_required": None, "step_therapy_detail": "", "step_therapy_citation": "",
        "quantity_limit_text": "", "quantity_limit_citation": "",
        "prescriber_specialty_required": [], "prescriber_specialty_citation": "",
        "prior_auth_required": None, "exclusions": [],
    }


def _diagnosis_prefix(diagnosis_code: str) -> str:
    """Pull the ICD-10 category (letter + 2 digits) off the front of a
    free-text diagnosis field like 'M06.9 (Rheumatoid arthritis, unspecified)'."""
    m = re.match(r"\s*([A-Za-z]\d{2})", diagnosis_code or "")
    return m.group(1).upper() if m else ""


def _first_number(text: str):
    m = re.search(r"\d+(\.\d+)?", text or "")
    return float(m.group(0)) if m else None


def _specialty_matches(required_specialty: str, prescriber_field_lower: str) -> bool:
    """Match a required specialty ('rheumatologist') against a free-text
    prescriber field ('Dr. Jane Alvarez, Rheumatology') without demanding an
    exact substring — plan docs and prescriber fields use different noun
    forms of the same specialty (rheumatologist / rheumatology) routinely."""
    spec = required_specialty.lower().strip()
    if spec in prescriber_field_lower:
        return True
    # Strip common profession-noun suffixes to get a shared stem, e.g.
    # "rheumatologist" -> "rheumatolog", which is also a prefix of
    # "rheumatology". Check both directions since either form could be shorter.
    for suffix in ("ist", "y"):
        if spec.endswith(suffix):
            stem = spec[: -len(suffix)]
            if len(stem) >= 5 and stem in prescriber_field_lower:
                return True
    return False


def check_criteria(criteria: dict, prescription: dict) -> list:
    """Deterministic pass — no LLM. Returns a list of checklist items, each
    with a status of 'met' / 'not_met' / 'needs_review' / 'not_applicable'.
    """
    checklist = []

    # 1. Diagnosis coverage
    prefixes = [p.strip().upper().rstrip(".-") for p in criteria["covered_icd10_prefixes"]]
    rx_prefix = _diagnosis_prefix(prescription.get("diagnosis_code", ""))
    if not prefixes:
        checklist.append({
            "criterion": "Diagnosis coverage",
            "status": "needs_review",
            "detail": "Plan document did not specify which ICD-10 codes are covered for this drug.",
            "citation": "",
        })
    elif not rx_prefix:
        checklist.append({
            "criterion": "Diagnosis coverage",
            "status": "needs_review",
            "detail": f"Could not parse an ICD-10 prefix from the submitted diagnosis code '{prescription.get('diagnosis_code', '')}'.",
            "citation": criteria["covered_icd10_citation"],
        })
    else:
        matched = any(rx_prefix.startswith(p) or p.startswith(rx_prefix) for p in prefixes)
        checklist.append({
            "criterion": "Diagnosis coverage",
            "status": "met" if matched else "not_met",
            "detail": (
                f"Diagnosis {rx_prefix} {'matches' if matched else 'does not match'} covered prefixes: {', '.join(prefixes)}."
            ),
            "citation": criteria["covered_icd10_citation"],
        })

    # 2. Step therapy
    st_required = criteria["step_therapy_required"]
    documented = prescription.get("step_therapy_documented")
    if st_required is None:
        checklist.append({
            "criterion": "Step therapy",
            "status": "needs_review",
            "detail": "Plan document did not address step therapy for this drug in the retrieved context.",
            "citation": criteria["step_therapy_citation"],
        })
    elif st_required is False:
        checklist.append({
            "criterion": "Step therapy",
            "status": "not_applicable",
            "detail": "Plan document does not require step therapy for this drug.",
            "citation": criteria["step_therapy_citation"],
        })
    else:  # required True
        if documented is True:
            checklist.append({
                "criterion": "Step therapy",
                "status": "met",
                "detail": f"Required by plan ({criteria['step_therapy_detail']}) and prescriber has documented it.",
                "citation": criteria["step_therapy_citation"],
            })
        elif documented is False:
            checklist.append({
                "criterion": "Step therapy",
                "status": "not_met",
                "detail": f"Required by plan ({criteria['step_therapy_detail']}) and prescriber indicated it is not documented.",
                "citation": criteria["step_therapy_citation"],
            })
        else:
            checklist.append({
                "criterion": "Step therapy",
                "status": "needs_review",
                "detail": f"Required by plan ({criteria['step_therapy_detail']}) — documentation status not provided with this request.",
                "citation": criteria["step_therapy_citation"],
            })

    # 3. Quantity limit — best-effort numeric comparison, fuzzy by nature.
    limit_text = criteria["quantity_limit_text"]
    rx_quantity = prescription.get("quantity") or ""
    if not limit_text:
        checklist.append({
            "criterion": "Quantity limit",
            "status": "needs_review",
            "detail": "Plan document did not specify a quantity limit for this drug in the retrieved context.",
            "citation": "",
        })
    else:
        limit_num = _first_number(limit_text)
        rx_num = _first_number(rx_quantity)
        if limit_num is not None and rx_num is not None:
            if rx_num <= limit_num:
                status, detail = "met", f"Requested quantity ({rx_quantity!r}) is within the plan limit ({limit_text!r})."
            else:
                status, detail = "not_met", f"Requested quantity ({rx_quantity!r}) appears to exceed the plan limit ({limit_text!r}) — dose-escalation documentation may be required."
        else:
            status, detail = "needs_review", f"Could not confidently compare requested quantity ({rx_quantity!r}) to plan limit ({limit_text!r}) by units — verify manually."
        checklist.append({
            "criterion": "Quantity limit",
            "status": status,
            "detail": detail,
            "citation": criteria["quantity_limit_citation"],
        })

    # 4. Prescriber specialty
    required_specialties = criteria["prescriber_specialty_required"]
    prescriber = (prescription.get("prescriber_name") or "").lower()
    if not required_specialties:
        checklist.append({
            "criterion": "Prescriber requirement",
            "status": "not_applicable",
            "detail": "Plan document does not specify a required prescriber specialty for this drug.",
            "citation": criteria["prescriber_specialty_citation"],
        })
    elif not prescriber:
        checklist.append({
            "criterion": "Prescriber requirement",
            "status": "needs_review",
            "detail": f"Plan requires one of: {', '.join(required_specialties)}. No prescriber field was submitted with this request.",
            "citation": criteria["prescriber_specialty_citation"],
        })
    else:
        matched = any(_specialty_matches(spec, prescriber) for spec in required_specialties)
        checklist.append({
            "criterion": "Prescriber requirement",
            "status": "met" if matched else "needs_review",
            "detail": (
                f"Prescriber field {'matches' if matched else 'does not clearly match'} one of the required specialties "
                f"({', '.join(required_specialties)})."
            ),
            "citation": criteria["prescriber_specialty_citation"],
        })

    return checklist
