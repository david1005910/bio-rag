import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager

from src.models import Base, engine
from src.api import auth_router, papers_router, chat_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("Database tables created")
    yield
    print("Shutting down")

app = FastAPI(
    title="Bio-RAG API",
    description="Biomedical Research AI-Guided Analytics Platform",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(papers_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "bio-rag"}

@app.get("/api/v1/stats")
async def get_stats():
    try:
        from src.services.vector_store import QdrantVectorStore
        vector_store = QdrantVectorStore(collection_name="biomedical_papers_768d", dimension=768)
        info = vector_store.get_collection_info()
        return {
            "indexed_chunks": info.get("points_count", 0),
            "collection": info.get("name", "biomedical_papers_768d")
        }
    except Exception as e:
        return {"indexed_chunks": 0, "error": str(e)}

@app.get("/{path:path}")
async def serve_frontend(path: str = ""):
    if path.startswith("api/"):
        return {"detail": "Not found"}
    
    file_path = f"static/{path}"
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
