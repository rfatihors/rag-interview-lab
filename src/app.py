from pathlib import Path

from fastapi import FastAPI, UploadFile

from src.config import settings
from src.core.schemas import AskRequest, AskResponse, IngestResponse
from src.providers.factory import build_provider
from src.rag.pipeline import build_pipeline
from src.vectorstores.factory import build_vector_store


app = FastAPI(title="RAG Interview Lab")


_provider = build_provider()
_vector_store = build_vector_store()
_pipeline = build_pipeline(_provider, _vector_store)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.llm_provider,
        "qdrant_url": settings.qdrant_url,
        "collection": settings.qdrant_collection,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    result = _pipeline.ingest(recreate=True)

    return {
        "status": "indexed",
        **result,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return _pipeline.ask(request.question)


@app.post("/upload")
async def upload_file(file: UploadFile):
    docs_dir = Path(settings.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    save_path = docs_dir / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    return {
        "status": "uploaded",
        "filename": file.filename,
        "path": str(save_path),
        "next_step": "POST /ingest çağırarak index'i güncelleyin.",
    }
