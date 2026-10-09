from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.utils.config import settings
from src.api.routes import router
from src.rag.hybrid_retriever import hybrid_retriever
from src.services.ml_models import claims_ml_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure retriever and ML models are pre-warmed
    try:
        print("[Startup] Initializing Hybrid Retriever and vector index...")
        hybrid_retriever.initialize()
        print("[Startup] Training/loading ML prediction models...")
        claims_ml_service.train_or_load()
        print("[Startup] All systems initialized successfully.")
    except Exception as e:
        print(f"[Startup Warning] Pre-warming encountered error: {e}")
    yield
    print("[Shutdown] Cleaning up resources...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="ClaimLens — AI-Powered Insurance Claims Assistant providing Hybrid RAG search, Multi-Agent workflow, ML risk predictions, and financial settlement governance.",
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for trusted local clients and Streamlit dashboard
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_PREFIX)
app.include_router(router, prefix="")  # Mount at root as well for flexibility

if __name__ == "__main__":
    uvicorn.run("src.main:app", host=settings.HOST, port=settings.PORT, reload=True)
