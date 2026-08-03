"""Tool implementation the drafting agent can call to re-query the indexed plan document."""
import rag

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_plan_document",
            "description": (
                "Semantically search the patient's indexed insurance plan document for "
                "clauses relevant to a query (e.g. step therapy requirements, quantity "
                "limits, a specific drug class, prior-authorization criteria). Use this "
                "when the initially retrieved context doesn't cover something you need "
                "to check before drafting the request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for in the plan document"},
                },
                "required": ["query"],
            },
        },
    },
]


def execute_tool_call(session: dict, name: str, args: dict) -> str:
    if name == "search_plan_document":
        query = args.get("query", "")
        results = rag.retrieve(session, query, top_k=4)
        if not results:
            return "No relevant clauses found."
        parts = []
        for r in results:
            parts.append(
                f"[{r['source']}:{r['start_line']}-{r['end_line']} score={r['score']:.3f}]\n{r['text']}"
            )
        return "\n\n".join(parts)
    return f"Unknown tool: {name}"
