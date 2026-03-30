"""
NUST Admission Helper - FastAPI Backend
Main application serving the chatbot API.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("NUSTApp")

# ─────────────────────────────────────────────
# Programs Database
# ─────────────────────────────────────────────

PROGRAMS_DATA = [
    {"name": "BS Computer Science", "school": "SEECS", "degree": "BS", "duration": "4yr", "seats": 250, "note": "Most competitive program at NUST"},
    {"name": "BS Artificial Intelligence", "school": "SEECS", "degree": "BS", "duration": "4yr", "seats": 50, "note": "New program with high industry demand"},
    {"name": "BE Computer Engineering", "school": "SEECS", "degree": "BE", "duration": "4yr", "seats": 50, "note": "Hardware + software integration"},
    {"name": "BS Software Engineering", "school": "SEECS", "degree": "BS", "duration": "4yr", "seats": 75, "note": "Industry-focused software development"},
    {"name": "BS Information Technology", "school": "SEECS", "degree": "BS", "duration": "4yr", "seats": 50, "note": "IT systems and networking"},
    {"name": "BE Electrical Engineering", "school": "SEECS", "degree": "BE", "duration": "4yr", "seats": 100, "note": "Power systems and electronics"},
    {"name": "BE Mechanical Engineering", "school": "SMME", "degree": "BE", "duration": "4yr", "seats": 120, "note": "Core engineering discipline"},
    {"name": "BE Mechatronics & Control Engineering", "school": "SMME", "degree": "BE", "duration": "4yr", "seats": 60, "note": "Robotics and automation"},
    {"name": "BE Industrial & Manufacturing Engineering", "school": "SMME", "degree": "BE", "duration": "4yr", "seats": 60, "note": "Production and operations management"},
    {"name": "BE Civil Engineering", "school": "SCEE", "degree": "BE", "duration": "4yr", "seats": 100, "note": "Infrastructure and construction"},
    {"name": "BE Environmental Engineering", "school": "SCEE", "degree": "BE", "duration": "4yr", "seats": 50, "note": "Environmental management and sustainability"},
    {"name": "BE Chemical Engineering", "school": "SCME", "degree": "BE", "duration": "4yr", "seats": 60, "note": "Process industries and petrochemicals"},
    {"name": "BS Materials Engineering", "school": "SCME", "degree": "BS", "duration": "4yr", "seats": 40, "note": "Advanced materials science"},
    {"name": "BBA", "school": "NBS", "degree": "BBA", "duration": "4yr", "seats": 120, "note": "Business administration and management"},
    {"name": "BS Accounting & Finance", "school": "NBS", "degree": "BS", "duration": "4yr", "seats": 80, "note": "Financial management and accounting"},
    {"name": "Bachelor of Architecture", "school": "SADA", "degree": "B.Arch", "duration": "5yr", "seats": 40, "note": "5-year professional architecture degree"},
    {"name": "BS Biotechnology", "school": "ASAB", "degree": "BS", "duration": "4yr", "seats": 50, "note": "Life sciences and biotech"},
    {"name": "BS Bioinformatics", "school": "ASAB", "degree": "BS", "duration": "4yr", "seats": 30, "note": "Computational biology"},
    {"name": "BS Microbiology", "school": "ASAB", "degree": "BS", "duration": "4yr", "seats": 30, "note": "Microbial sciences"},
    {"name": "BS Mathematics", "school": "SNS", "degree": "BS", "duration": "4yr", "seats": 40, "note": "Pure and applied mathematics"},
    {"name": "BS Physics", "school": "SNS", "degree": "BS", "duration": "4yr", "seats": 40, "note": "Theoretical and experimental physics"},
    {"name": "BS Chemistry", "school": "SNS", "degree": "BS", "duration": "4yr", "seats": 40, "note": "Chemical sciences"},
    {"name": "BS Economics", "school": "S3H", "degree": "BS", "duration": "4yr", "seats": 60, "note": "Economic theory and policy"},
    {"name": "BS Psychology", "school": "S3H", "degree": "BS", "duration": "4yr", "seats": 50, "note": "Behavioral and cognitive science"},
    {"name": "BS Development Studies", "school": "S3H", "degree": "BS", "duration": "4yr", "seats": 40, "note": "Social development and policy"},
    {"name": "BS Geographical Information Systems", "school": "IGIS", "degree": "BS", "duration": "4yr", "seats": 30, "note": "GIS, remote sensing, spatial data"},
]

# Load merit cutoffs from the permanent cutoffs.json (never overwritten by scraper)
def _load_cutoffs() -> dict:
    cutoffs_file = ROOT / "data" / "cutoffs.json"
    with open(cutoffs_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["name"]: item["cutoff"] for item in data["cutoffs"]}

CUTOFFS = _load_cutoffs()


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    session: dict = {}


class MeritRequest(BaseModel):
    net_score: float
    fsc_marks: float
    fsc_total: float = 1100.0
    matric_marks: float
    matric_total: float = 1100.0
    has_hafiz: bool = False  # kept for API compatibility but ignored


# ─────────────────────────────────────────────
# Lifespan - Startup/Shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize chain and dependencies on startup."""
    logger.info("Starting NUST Admission Helper backend...")

    # Import chain (lazy initialization)
    try:
        from chatbot.chain import NUSTChain
        from chatbot.llm import OllamaLLM
        from rag.retriever import NUSTRetriever

        app.state.llm = OllamaLLM()
        app.state.retriever = NUSTRetriever()
        app.state.chain = NUSTChain(llm=app.state.llm, retriever=app.state.retriever)
        logger.info("Chain initialized successfully.")
    except Exception as e:
        logger.error("Failed to initialize chain: %s", e)
        app.state.chain = None
        app.state.llm = None
        app.state.retriever = None

    # Import scraper status
    try:
        from scraper.scrape import NUSTScraper
        app.state.scraper = NUSTScraper()
        # Populate status from existing files
        app.state.scraper.get_scraper_status()
    except Exception as e:
        logger.warning("Scraper init failed: %s", e)
        app.state.scraper = None

    yield

    logger.info("Shutting down NUST Admission Helper backend.")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="NUST Admission Helper API",
    description="AI-powered NUST admission chatbot backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_HTML = ROOT / "index.html"


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
async def serve_index():
    """Serve the main index.html file."""
    if INDEX_HTML.exists():
        return FileResponse(str(INDEX_HTML), media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/api/health")
async def health_check(request: Request):
    """
    Returns system health status:
    - Ollama status and model
    - RAG readiness
    - Scraper status
    """
    ollama_status = "offline"
    model_name = os.getenv("MODEL", "phi3:mini")
    rag_ready = False
    scraper_status = {"ready": False, "pages_scraped": 0, "last_refresh": "Never"}

    # Check Ollama
    if hasattr(request.app.state, "llm") and request.app.state.llm:
        try:
            health = await request.app.state.llm.check_ollama_health()
            ollama_status = health.get("status", "offline")
            model_name = health.get("model", model_name)
        except Exception as e:
            logger.warning("Health check Ollama error: %s", e)
    else:
        # Fallback: try to connect directly
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get("http://localhost:11434/api/tags")
                if resp.status_code == 200:
                    ollama_status = "running"
        except Exception:
            pass

    # Check RAG
    if hasattr(request.app.state, "retriever") and request.app.state.retriever:
        try:
            rag_ready = request.app.state.retriever.is_ready()
        except Exception:
            rag_ready = False

    # Check scraper
    if hasattr(request.app.state, "scraper") and request.app.state.scraper:
        try:
            scraper_status = request.app.state.scraper.get_scraper_status()
        except Exception:
            pass

    return {
        "ollama": ollama_status,
        "rag_ready": rag_ready,
        "model": model_name,
        "live_scraper": scraper_status,
    }


@app.post("/api/chat")
async def chat(request_data: ChatRequest, request: Request):
    """
    SSE streaming chat endpoint.
    Streams response tokens in format: data: {"t": "token"}\n\n
    """
    msg = request_data.message.strip()
    history = request_data.history
    session = request_data.session

    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    chain = getattr(request.app.state, "chain", None)

    async def generate():
        if chain is None:
            # No chain available — return fallback
            fallback = (
                "I'm having trouble initializing. Please ensure all dependencies are installed "
                "and Ollama is running. Check the README for setup instructions."
            )
            yield f"data: {json.dumps({'t': fallback})}\n\n"
            yield "data: [DONE]\n\n"
            return

        try:
            async for token in chain.stream(msg, history, session):
                if token:
                    yield f"data: {json.dumps({'t': token})}\n\n"
        except Exception as e:
            logger.error("Streaming error: %s", e)
            error_msg = f"An error occurred: {str(e)}"
            yield f"data: {json.dumps({'t': error_msg})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/programs")
async def get_programs():
    """Return list of all NUST programs."""
    return {"programs": PROGRAMS_DATA}


@app.post("/api/merit")
async def calculate_merit(data: MeritRequest):
    """
    Calculate NUST aggregate and program eligibility.

    Formula:
    Aggregate = (NET/200 * 75) + (FSc/FSc_Total * 15) + (Matric/Matric_Total * 10)
    """
    # Validate inputs
    if data.net_score < 0 or data.net_score > 200:
        raise HTTPException(status_code=400, detail="NET score must be between 0 and 200")
    if data.fsc_marks < 0 or data.fsc_marks > data.fsc_total:
        raise HTTPException(status_code=400, detail="FSc marks cannot exceed FSc total")
    if data.matric_marks < 0 or data.matric_marks > data.matric_total:
        raise HTTPException(status_code=400, detail="Matric marks cannot exceed Matric total")
    if data.fsc_total <= 0 or data.matric_total <= 0:
        raise HTTPException(status_code=400, detail="Totals must be greater than 0")

    effective_net = data.net_score  # no bonus marks exist in NUST admissions

    # Calculate contributions
    net_c = (effective_net / 200) * 75
    fsc_c = (data.fsc_marks / data.fsc_total) * 15
    mat_c = (data.matric_marks / data.matric_total) * 10

    aggregate = round(net_c + fsc_c + mat_c, 2)

    # Determine rating
    if aggregate >= 90:
        rating = "Excellent - Top Programs"
    elif aggregate >= 85:
        rating = "Very Strong"
    elif aggregate >= 80:
        rating = "Strong"
    elif aggregate >= 75:
        rating = "Good"
    elif aggregate >= 70:
        rating = "Average"
    elif aggregate >= 65:
        rating = "Below Average"
    else:
        rating = "Needs Improvement"

    # Determine eligible programs
    eligible_programs = []
    close_programs = []

    for prog in PROGRAMS_DATA:
        name = prog["name"]
        cutoff = CUTOFFS.get(name)
        if cutoff is None:
            continue

        if aggregate >= cutoff:
            status = "Likely" if aggregate >= cutoff + 2 else "Borderline"
            eligible_programs.append({
                "name": name,
                "school": prog["school"],
                "cutoff": cutoff,
                "your_aggregate": aggregate,
                "margin": round(aggregate - cutoff, 1),
                "status": status,
            })
        elif cutoff - aggregate <= 5:
            close_programs.append({
                "name": name,
                "school": prog["school"],
                "cutoff": cutoff,
                "your_aggregate": aggregate,
                "needed": round(cutoff - aggregate, 1),
            })

    # Sort eligible by margin (descending), close by needed (ascending)
    eligible_programs.sort(key=lambda x: x["margin"], reverse=True)
    close_programs.sort(key=lambda x: x["needed"])

    # Generate contextual advice
    advice = _generate_advice(aggregate, eligible_programs, close_programs, data)

    return {
        "aggregate": aggregate,
        "breakdown": {
            "net_contribution": round(net_c, 2),
            "fsc_contribution": round(fsc_c, 2),
            "matric_contribution": round(mat_c, 2),
        },
        "effective_net": effective_net,
        "hafiz_bonus_applied": False,
        "rating": rating,
        "advice": advice,
        "eligible_programs": eligible_programs,
        "close_programs": close_programs,
    }


def _generate_advice(
    aggregate: float,
    eligible: list[dict],
    close: list[dict],
    data: MeritRequest,
) -> str:
    """Generate contextual admission advice based on aggregate score."""
    lines = []

    if aggregate >= 89:
        lines.append(
            f"Excellent aggregate of {aggregate}%! You have a strong chance at NUST's most competitive programs."
        )
        top = [p["name"] for p in eligible[:3]]
        if top:
            lines.append(f"Top choices: {', '.join(top)}.")
        lines.append("Focus on securing your desired program with a high NET score.")

    elif aggregate >= 80:
        lines.append(
            f"Strong aggregate of {aggregate}%. You're eligible for several excellent programs."
        )
        top = [p["name"] for p in eligible[:4]]
        if top:
            lines.append(f"Eligible programs include: {', '.join(top)}.")
        if close:
            close_names = [p["name"] for p in close[:2]]
            lines.append(
                f"With {close[0]['needed']}% more aggregate, you could also target: {', '.join(close_names)}."
            )

    elif aggregate >= 70:
        lines.append(
            f"Moderate aggregate of {aggregate}%. You qualify for several programs."
        )
        if eligible:
            top = [p["name"] for p in eligible[:3]]
            lines.append(f"Consider: {', '.join(top)}.")
        if close:
            lines.append(
                f"Improve your NET score to target programs like {close[0]['name']} "
                f"(need {close[0]['needed']}% more)."
            )
        lines.append(
            "Focus on improving your NET score — it contributes 50% to the aggregate."
        )

    else:
        lines.append(
            f"Your current aggregate of {aggregate}% may limit your choices at NUST."
        )
        if eligible:
            lines.append(f"You may be eligible for: {', '.join([p['name'] for p in eligible])}.")
        lines.append(
            "Prepare intensively for NET — improving your NET score has the biggest impact on your aggregate. "
            "Consider NUST's preparatory programs or other universities while you improve."
        )

    # NET-specific tip
    current_net = data.net_score
    if current_net < 160:
        needed_for_5_more = ((current_net + 20) / 200) * 50 - (current_net / 200) * 50
        lines.append(
            f"Tip: Every 20 NET marks adds approximately {needed_for_5_more:.1f}% to your aggregate."
        )


    return " ".join(lines)


@app.post("/api/refresh")
async def refresh_scraper(request: Request):
    """
    Trigger a fresh scrape of NUST website and update the knowledge base.
    Returns the scrape summary.
    """
    scraper = getattr(request.app.state, "scraper", None)
    if scraper is None:
        raise HTTPException(status_code=503, detail="Scraper not initialized")

    try:
        from scraper.scrape import NUSTScraper
        from scraper.cleaner import NUSTCleaner
        from rag.embedder import ChromaDBManager

        # Run scrape
        s = NUSTScraper()
        scrape_result = s.scrape_all()
        request.app.state.scraper = s

        # Clean the data
        cleaner = NUSTCleaner()
        clean_result = cleaner.clean_all()

        # Load and embed new chunks
        if clean_result.get("chunks_created", 0) > 0:
            chunks = cleaner.load_cleaned_chunks()
            db = ChromaDBManager()
            docs_added = db.upsert_documents(chunks)

            # Update retriever
            if hasattr(request.app.state, "retriever") and request.app.state.retriever:
                request.app.state.retriever.db = db

            return {
                "status": "success",
                "scrape": scrape_result,
                "clean": clean_result,
                "documents_added": docs_added,
            }

        return {
            "status": "partial",
            "scrape": scrape_result,
            "clean": clean_result,
            "message": "Scrape completed but no new chunks generated. Live site may be unreachable.",
        }

    except Exception as e:
        logger.error("Refresh error: %s", e)
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")
