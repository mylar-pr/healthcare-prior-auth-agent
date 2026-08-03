from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


# Real logic goes here: request/response models, business logic,
# integrations with external services, and any RAG/agent pipeline
# specific to this project would be wired up in this module or
# imported from additional modules alongside it.
