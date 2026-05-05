"""
FastAPI Application — Autonomous Data Analyst Agent

Endpoints
---------
POST /analyze          — Upload CSV + query → run full multi-agent analysis
GET  /sessions         — List all completed analysis sessions
GET  /sessions/{id}    — Retrieve a specific session result
GET  /health           — Health check
GET  /visualizations/{filename} — Serve saved plot images
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "agent.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Autonomous Data Analyst Agent",
    description=(
        "A production-grade multi-agent system that autonomously analyses CSV datasets "
        "using a Planner → Analyst → Critic pipeline powered by Claude."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory job store (use Redis/DB for production)
_jobs: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    query: str = Field(..., description="Natural language analysis query", min_length=5)
    max_retries: Optional[int] = Field(default=None, ge=0, le=5)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_plan_steps: Optional[int] = Field(default=None, ge=2, le=12)


class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | completed | failed
    message: Optional[str] = None
    progress: Optional[int] = None


class AnalysisResult(BaseModel):
    job_id: str
    session_id: str
    status: str
    elapsed_seconds: Optional[float] = None
    dataset_info: Dict[str, Any]
    analysis_plan: List[str]
    steps_summary: List[Dict[str, Any]]
    final_insights: str
    visualizations: List[str]
    session_file: Optional[str] = None


# ---------------------------------------------------------------------------
# Background analysis task
# ---------------------------------------------------------------------------

def _run_analysis_job(
    job_id: str,
    dataset_path: str,
    query: str,
    max_retries: int,
    score_threshold: float,
    max_plan_steps: int,
) -> None:
    """Background task: run the orchestrator and update job store."""
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["started_at"] = time.time()

    try:
        from core.orchestrator import Orchestrator

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")

        orchestrator = Orchestrator(
            groq_api_key=api_key,
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            max_retries=max_retries,
            score_threshold=score_threshold,
            max_plan_steps=max_plan_steps,
            output_dir=OUTPUT_DIR,
            log_dir=LOG_DIR,
        )

        result = orchestrator.run(dataset_path=dataset_path, query=query)
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
    finally:
        # Clean up uploaded temp file
        try:
            if os.path.exists(dataset_path) and "tmp" in dataset_path:
                os.remove(dataset_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check() -> JSONResponse:
    """Confirm the API is running and the API key is configured."""
    api_key_set = bool(os.getenv("GROQ_API_KEY"))
    return JSONResponse(
        {
            "status": "healthy",
            "api_key_configured": api_key_set,
            "output_dir": OUTPUT_DIR,
            "active_jobs": sum(1 for j in _jobs.values() if j["status"] == "running"),
        }
    )


@app.post("/analyze", tags=["Analysis"], response_model=JobStatus)
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="CSV dataset to analyse"),
    query: str = Form(..., description="Natural language query"),
    max_retries: int = Form(default=3),
    score_threshold: float = Form(default=0.65),
    max_plan_steps: int = Form(default=8),
) -> JobStatus:
    """
    Upload a CSV file and a natural language query.
    Returns a job_id immediately; analysis runs in the background.
    Poll /jobs/{job_id} for status and results.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith((".csv", ".tsv", ".docx")):
        raise HTTPException(status_code=400, detail="Only CSV, TSV, or DOCX files are accepted.")

    # Save uploaded file to a temp location
    job_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    if not ext:
        ext = ".csv"
    tmp_path = os.path.join(tempfile.gettempdir(), f"agent_{job_id}{ext}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    _jobs[job_id] = {
        "status": "queued",
        "query": query,
        "filename": file.filename,
        "dataset_path": tmp_path,
        "created_at": time.time(),
    }

    background_tasks.add_task(
        _run_analysis_job,
        job_id=job_id,
        dataset_path=tmp_path,
        query=query,
        max_retries=max_retries,
        score_threshold=score_threshold,
        max_plan_steps=max_plan_steps,
    )

    logger.info(f"Job {job_id} queued | file={file.filename} | query={query[:80]}")

    return JobStatus(
        job_id=job_id,
        status="queued",
        message="Analysis queued. Poll /jobs/{job_id} for updates.",
    )


@app.get("/jobs/{job_id}", tags=["Analysis"])
async def get_job_status(job_id: str) -> JSONResponse:
    """Check the status of a running or completed analysis job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job = _jobs[job_id]
    status = job["status"]

    response: Dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "query": job.get("query", ""),
        "filename": job.get("filename", ""),
    }

    if status == "running":
        elapsed = round(time.time() - job.get("started_at", time.time()), 1)
        response["elapsed_seconds"] = elapsed

    elif status == "completed":
        result = job.get("result", {})
        response.update(
            {
                "elapsed_seconds": result.get("elapsed_seconds"),
                "session_id": result.get("session_id"),
                "dataset_info": result.get("dataset_metadata", {}).get("shape", {}),
                "analysis_plan": result.get("analysis_plan", []),
                "steps_summary": [
                    {
                        "step_id": s["step_id"],
                        "description": s["description"],
                        "status": s["status"],
                        "critic_score": s["critic_score"],
                        "retry_count": s["retry_count"],
                        "output_preview": s["output"][:400],
                        "visualizations": s["visualizations"],
                    }
                    for s in result.get("steps", [])
                ],
                "final_insights": result.get("final_insights", ""),
                "visualizations": result.get("all_visualizations", []),
                "session_file": result.get("session_file"),
                "summary": result.get("summary", {}),
            }
        )

    elif status == "failed":
        response["error"] = job.get("error", "Unknown error")

    return JSONResponse(response)


@app.get("/sessions", tags=["Sessions"])
async def list_sessions() -> JSONResponse:
    """List all session JSON files saved in the output directory."""
    sessions = []
    for fname in sorted(Path(OUTPUT_DIR).glob("session_*.json"), reverse=True):
        try:
            with open(fname) as f:
                data = json.load(f)
            sessions.append(
                {
                    "session_id": data.get("session_id"),
                    "created_at": data.get("created_at"),
                    "user_query": data.get("user_query"),
                    "dataset": data.get("dataset_path"),
                    "steps_completed": data.get("summary", {}).get("completed", 0),
                    "avg_critic_score": data.get("summary", {}).get("avg_critic_score", 0),
                    "visualizations": len(data.get("all_visualizations", [])),
                    "file": str(fname),
                }
            )
        except Exception:
            continue
    return JSONResponse({"sessions": sessions, "total": len(sessions)})


def sanitize_nan(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nan(item) for item in obj]
    return obj


@app.get("/sessions/{session_id}", tags=["Sessions"])
async def get_session(session_id: str) -> JSONResponse:
    """Retrieve a specific session result by session ID."""
    path = Path(OUTPUT_DIR) / f"session_{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    with open(path) as f:
        data = json.load(f)
    return JSONResponse(sanitize_nan(data))


@app.get("/visualizations/{filename}", tags=["Outputs"])
async def serve_visualization(filename: str) -> FileResponse:
    """Download a saved visualization image."""
    # Security: prevent path traversal
    safe_name = Path(filename).name
    path = Path(OUTPUT_DIR) / safe_name
    if not path.exists() or not safe_name.endswith(".png"):
        raise HTTPException(status_code=404, detail="Visualization not found.")
    return FileResponse(str(path), media_type="image/png")


@app.delete("/jobs/{job_id}", tags=["Analysis"])
async def delete_job(job_id: str) -> JSONResponse:
    """Remove a job from the in-memory store."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    del _jobs[job_id]
    return JSONResponse({"deleted": job_id})


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

# Mount static files for the UI
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
