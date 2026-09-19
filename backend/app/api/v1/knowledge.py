from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.services.knowledge_ingestion import (
    knowledge_ingestion_service,
    SUPPORTED_EXTENSIONS,
)
from app.services.vector_store import vector_store

router = APIRouter()


class IngestKnowledgeRequest(BaseModel):
    text: str
    source: str


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 4


@router.post("/ingest")
async def ingest_knowledge(request: IngestKnowledgeRequest):
    """Index already-extracted trusted text into the local knowledge base."""
    try:
        chunk_count = await vector_store.add_document(
            request.text,
            request.source,
        )
        return {
            "status": "INGESTED",
            "source": request.source,
            "chunk_count": chunk_count,
            "collection": "company_knowledge",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge ingest failed: {exc}",
        )


@router.post("/ingest-file")
async def ingest_knowledge_file(
    file: UploadFile = File(...),
    source: Optional[str] = Form(None),
):
    """Persist and index one trusted organization document.

    This endpoint is intentionally separate from /files/purify:
    ordinary user attachments must not silently become organizational knowledge.
    """
    filename = Path(file.filename or "knowledge_document").name
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported knowledge file type: {ext}",
        )

    target = settings.KNOWLEDGE_DIR / filename

    # Avoid path traversal and accidental writes outside the knowledge directory.
    if target.resolve().parent != settings.KNOWLEDGE_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        content = await file.read()
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

        if len(content) > max_bytes:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit"
                ),
            )

        target.write_bytes(content)

        return await knowledge_ingestion_service.ingest_path(
            target,
            source=source or filename,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge file ingest failed: {exc}",
        )


@router.post("/ingest-directory")
async def ingest_knowledge_directory():
    """Index all supported files placed in data/knowledge_documents."""
    try:
        results = await knowledge_ingestion_service.ingest_directory()
        return {
            "status": "REBUILT",
            "results": results,
            "total_chunks": await vector_store.count(),
            "collection": "company_knowledge",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge directory ingest failed: {exc}",
        )


@router.post("/search")
async def search_knowledge(request: KnowledgeSearchRequest):
    """Debug/evaluation endpoint for inspecting retrieval quality."""
    try:
        results = await knowledge_ingestion_service.search(
            request.query,
            top_k=max(1, min(request.top_k, 10)),
        )
        return {
            "query": request.query,
            "results": results,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge search failed: {exc}",
        )


@router.get("/status")
async def knowledge_status():
    return {
        "collection": "company_knowledge",
        "documents_chunks": await vector_store.count(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "storage": str(settings.VECTOR_DB_PATH),
        "knowledge_directory": str(settings.KNOWLEDGE_DIR),
    }
