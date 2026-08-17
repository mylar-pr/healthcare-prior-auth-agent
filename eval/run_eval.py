#!/usr/bin/env python3
"""Eval harness for the criteria checklist pipeline.

Runs each case in cases.json through the real pipeline — build_index ->
retrieve -> extract_criteria (real MiniMax call) -> check_criteria (pure
Python) — and asserts the checklist statuses match what's expected. This is
the "prove the pattern generalizes past one document" harness: each case
uses a differently-structured synthetic plan document and a prescription
chosen to exercise a specific outcome (covered, not covered, step-therapy
exempt, quantity exceeded, etc).

Usage:
    cd backend && ../.venv/bin/python ../eval/run_eval.py
    (or from repo root: backend/.venv/bin/python eval/run_eval.py)

Exits non-zero if any case fails, so it's CI-friendly.
"""
import json
import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(BACKEND_DIR, ".env"))

import criteria as criteria_mod  # noqa: E402
import rag  # noqa: E402

EVAL_DIR = os.path.dirname(__file__)


def load_cases():
    with open(os.path.join(EVAL_DIR, "cases.json")) as f:
        return json.load(f)


def run_case(case: dict) -> tuple:
    plan_path = os.path.join(EVAL_DIR, case["plan_file"])
    with open(plan_path, encoding="utf-8") as f:
        plan_text = f.read()

    session = rag.build_index(plan_text, source_name=os.path.basename(plan_path))

    prescription = case["prescription"]
    query = f"{prescription['drug_name']} {prescription['dosage']} {prescription['diagnosis_code']} prior authorization coverage criteria"
    boost_terms = [prescription["drug_name"], prescription["diagnosis_code"]]
    retrieved = rag.retrieve(session, query, top_k=6, boost_terms=boost_terms)

    extracted = criteria_mod.extract_criteria(retrieved)
    checklist = criteria_mod.check_criteria(extracted, prescription)
    actual = {item["criterion"]: item["status"] for item in checklist}

    failures = []
    for criterion, expected_status in case["expected"].items():
        actual_status = actual.get(criterion, "<missing>")
        if actual_status != expected_status:
            failures.append(f"    {criterion}: expected {expected_status!r}, got {actual_status!r}")

    return failures, checklist


def main():
    if not os.environ.get("MINIMAX_API_KEY"):
        print("MINIMAX_API_KEY not set (check backend/.env) — cannot run eval.")
        sys.exit(1)

    cases = load_cases()
    total_failures = 0

    for case in cases:
        print(f"\n=== {case['name']} ===")
        print(f"    {case['description']}")
        try:
            failures, checklist = run_case(case)
        except Exception as e:  # noqa: BLE001 - eval harness, want to keep going and report
            print(f"  ERROR running case: {e}")
            total_failures += 1
            continue

        if failures:
            print("  FAIL")
            for line in failures:
                print(line)
            print("  Full checklist:")
            for item in checklist:
                print(f"    - {item['criterion']}: {item['status']} — {item['detail']}")
            total_failures += len(failures)
        else:
            print("  PASS —", ", ".join(f"{k}={v}" for k, v in case["expected"].items()))

    print(f"\n{'='*60}")
    if total_failures:
        print(f"{total_failures} check(s) failed across {len(cases)} case(s).")
        sys.exit(1)
    else:
        print(f"All checks passed across {len(cases)} case(s).")


if __name__ == "__main__":
    main()
