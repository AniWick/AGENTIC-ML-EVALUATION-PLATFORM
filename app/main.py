import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.database import db
from app.api import projects, models, datasets, evaluations, experiments, chat, reports, presets

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Agentic ML Model & Dataset Evaluation Platform"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(projects.router)
app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(evaluations.router)
app.include_router(experiments.router)
app.include_router(chat.router)
app.include_router(reports.router)
app.include_router(presets.router)

# Mount Frontend
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
def serve_frontend():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Agentic ML Model & Dataset Evaluation Platform Backend API"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "llm_model": settings.LLM_MODEL,
        "ollama_url": settings.OLLAMA_BASE_URL
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

