from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.network_guard import install_network_guard

install_network_guard()

from app.api.v1.agent import router as agent_router
from app.api.v1.ingest import router as ingest_router, purifier
from app.api.v1.knowledge import router as knowledge_router
from app.api.v1.router import router as task_router
from app.api.v1.system import router as system_router
from app.api.v1.speech import router as speech_router
from app.services.components.image_component import image_component
from app.services.vector_store import vector_store

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

@app.on_event("startup")
async def warm_up_models():
    # Loads PaddleOCR at server startup instead of paying that cost on the first request
    _ = image_component.ocr_engine
    # Purges stale workspace files left over from prior sessions
    purifier.cleanup_workspace()

    # Warm the local embedding model once so the first RAG query does not pay
    # the model-load cost. Failure is non-fatal: the first RAG request can retry.
    try:
        await vector_store._embed("local knowledge base warmup")
    except Exception:
        # Ollama may not be running yet during application startup.
        pass

# Allow the local intranet only (localhost + private LAN ranges), never public internet origins
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route Wiring
app.include_router(ingest_router, prefix=f"{settings.API_V1_STR}/files", tags=["File Ingestion"])
app.include_router(task_router, prefix=f"{settings.API_V1_STR}/tasks", tags=["Task Routing"])
app.include_router(agent_router, prefix=f"{settings.API_V1_STR}/agent", tags=["Agent Orchestrator"])
app.include_router(knowledge_router, prefix=f"{settings.API_V1_STR}/knowledge", tags=["Knowledge Base"])
app.include_router(system_router, prefix=f"{settings.API_V1_STR}/system", tags=["System"])
app.include_router(speech_router, prefix=f"{settings.API_V1_STR}/speech", tags=["Speech-to-Text"])

@app.get("/health")
def health_check():
    return {"status": "online", "mode": "air-gapped", "sovereignty": "verified"}