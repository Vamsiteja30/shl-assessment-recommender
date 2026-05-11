import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import init_agent, get_agent
from app.conversation import process_conversation
from app.retriever import init_retriever, get_retriever
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup and clean up on shutdown."""
    logger.info("Service starting...")

    index_path = os.environ.get("FAISS_INDEX_PATH", "data/faiss_index.bin")
    meta_path = os.environ.get("FAISS_META_PATH", "data/faiss_meta.pkl")

    if not Path(index_path).exists() or not Path(meta_path).exists():
        logger.error(f"FAISS index files missing at {index_path} or {meta_path}")
        raise RuntimeError("Index files not found. Run catalog/build_index.py first.")

    init_retriever(index_path, meta_path)
    init_agent()

    logger.info("Retriever and agent initialized.")
    yield
    logger.info("Service shutting down.")

app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a schema-compliant response on unhandled errors."""
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=200,
        content={
            "reply": "I'm having some trouble processing that right now. Could you tell me more about the role or skills you're looking for?",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )

@app.get("/")
async def root():
    return {
        "service": "SHL Assessment Recommender",
        "endpoints": ["/health", "/chat"]
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Main entry point for the conversational recommender."""
    
    # Enforce conversation turn limit
    if len(request.messages) > 8:
        return ChatResponse(
            reply="I've shared my best recommendations for this role. Is there anything else you'd like to know about our assessments?",
            recommendations=[],
            end_of_conversation=True,
        )

    retriever = get_retriever()
    agent = get_agent()

    return await process_conversation(request, retriever, agent)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
