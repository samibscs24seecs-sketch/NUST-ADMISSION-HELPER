"""
NUST Admission Helper - Main Entry Point
Starts the FastAPI backend with all required checks.
"""

import json
import logging
import os
import sys
import time
import webbrowser
from pathlib import Path

# ─────────────────────────────────────────────
# Project root setup
# ─────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "data" / "app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("NUSTMain")

# ─────────────────────────────────────────────
# ASCII Banner
# ─────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    ███╗   ██╗██╗   ██╗███████╗████████╗                     ║
║    ████╗  ██║██║   ██║██╔════╝╚══██╔══╝                     ║
║    ██╔██╗ ██║██║   ██║███████╗   ██║                        ║
║    ██║╚██╗██║██║   ██║╚════██║   ██║                        ║
║    ██║ ╚████║╚██████╔╝███████║   ██║                        ║
║    ╚═╝  ╚═══╝ ╚═════╝ ╚══════╝   ╚═╝                        ║
║                                                              ║
║         NUST Admission Helper Chatbot v1.0                   ║
║         NUST Local Chatbot Competition 2026                  ║
║         Powered by Ollama + ChromaDB + FastAPI               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "127.0.0.1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL", "llama3.2:3b")


def print_banner():
    print(BANNER)


def create_directories():
    """Create all required project directories."""
    dirs = [
        ROOT / "data" / "raw",
        ROOT / "data" / "cleaned",
        ROOT / "database" / "chroma_store",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("All directories verified/created.")


def check_ram():
    """Display current RAM usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        used_gb = mem.used / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        percent = mem.percent

        print(f"\n{'─'*60}")
        print(f"  System RAM: {total_gb:.1f} GB total | {used_gb:.1f} GB used | {available_gb:.1f} GB free")
        print(f"  RAM Usage: {percent}%", end="")

        if available_gb < 2.0:
            print(" ⚠️  WARNING: Low RAM — LLM may run slowly")
        elif available_gb < 4.0:
            print(" ⚠️  Moderate RAM available")
        else:
            print(" ✓  Sufficient RAM")
        print(f"{'─'*60}")

    except ImportError:
        print("  psutil not installed — RAM check skipped")
    except Exception as e:
        print(f"  RAM check error: {e}")


def check_ollama() -> dict:
    """Check if Ollama is running and llama3.2:3b is available."""
    import urllib.request
    import urllib.error

    print(f"\n{'─'*60}")
    print("  Checking Ollama...")

    try:
        req = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5)
        data = json.loads(req.read().decode())
        models = [m["name"] for m in data.get("models", [])]

        print(f"  ✓ Ollama is running at {OLLAMA_URL}")
        print(f"  Available models: {', '.join(models) if models else 'None'}")

        if not models:
            print(f"\n  ✗ No models downloaded yet.")
            print(f"  Run: ollama pull llama3.2:3b")
            return {"running": True, "model_ready": False, "models": models}

        # llama3.2:3b — ~2 GB RAM (Q4). Safe on 8 GB CPU-only systems.
        model_ready = any("llama3.2:3b" in m for m in models)

        if model_ready:
            actual = next(m for m in models if "llama3.2:3b" in m)
            print(f"  ✓ Model ready: {actual}")
            os.environ["MODEL"] = actual
        else:
            print(f"\n  ✗ llama3.2:3b not found.")
            print(f"  Run: ollama pull llama3.2:3b")

        return {"running": True, "model_ready": model_ready, "models": models}

    except urllib.error.URLError:
        print(f"  ✗ Ollama is NOT running at {OLLAMA_URL}")
        print(f"\n  To fix: Open a terminal and run: ollama serve")
        print(f"  Then pull a model: ollama pull {MODEL_NAME}")
        print(f"\n  ℹ️  The chatbot will start but AI responses will be unavailable")
        print(f"     until Ollama is running.")
        print(f"{'─'*60}")
        return {"running": False, "model_ready": False, "models": []}
    except Exception as e:
        print(f"  ✗ Unexpected Ollama check error: {e}")
        return {"running": False, "model_ready": False, "models": []}


def check_index_html():
    """Check if index.html exists."""
    index_path = ROOT / "index.html"
    print(f"\n{'─'*60}")
    if index_path.exists():
        size = index_path.stat().st_size / 1024
        print(f"  ✓ index.html found ({size:.1f} KB)")
        return True
    else:
        print(f"  ✗ index.html NOT FOUND at {index_path}")
        print(f"  Please place index.html in the project root: {ROOT}")
        return False


def load_static_knowledge():
    """Load static knowledge base into ChromaDB if it's empty."""
    knowledge_file = ROOT / "data" / "nust_knowledge.json"

    if not knowledge_file.exists():
        print(f"  ✗ Knowledge base file not found: {knowledge_file}")
        return False

    print(f"\n{'─'*60}")
    print("  Checking knowledge base (ChromaDB)...")

    try:
        from rag.embedder import ChromaDBManager

        # Check count WITHOUT loading the embedding model — avoids sentence-transformer
        # initialization and HuggingFace network calls on every startup.
        existing = ChromaDBManager.quick_count()
        if existing > 0:
            print(f"  ✓ ChromaDB ready — {existing} entries already loaded (skipping re-embed)")
            return True

        db = ChromaDBManager()

        # First run only: embed all entries from nust_knowledge.json
        print(f"  First run — embedding knowledge base (takes 2-3 min on first launch)...")
        print(f"  ⏳ Downloading sentence-transformer model if needed...")

        with open(knowledge_file, encoding="utf-8") as f:
            entries = json.load(f)

        from datetime import datetime
        now = datetime.now().isoformat()
        chunks = []
        for entry in entries:
            chunks.append({
                "chunk_id": entry.get("id", f"static_{len(chunks):04d}"),
                "text": entry.get("text", ""),
                "source_url": entry.get("source", "https://nust.edu.pk/"),
                "scraped_at": now,
                "category": entry.get("category", "general"),
            })

        upserted = db.upsert_documents(chunks)
        print(f"  ✓ ChromaDB ready — {upserted} entries embedded")
        return True

    except ImportError as e:
        print(f"  ⚠️  Cannot load ChromaDB: {e}")
        print(f"  Run: pip install chromadb sentence-transformers")
        return False
    except Exception as e:
        logger.error("Knowledge base loading error: %s", e)
        print(f"  ✗ Failed to load knowledge base: {e}")
        return False


def open_browser_delayed(url: str, delay: float = 2.0):
    """Open browser after a short delay to allow server to start."""
    import threading

    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
            logger.info("Browser opened: %s", url)
        except Exception as e:
            logger.warning("Could not open browser: %s", e)

    t = threading.Thread(target=_open, daemon=True)
    t.start()


def print_startup_summary(ollama_status: dict, rag_ready: bool):
    """Print a summary of system status before starting."""
    url = f"http://{HOST}:{PORT}"
    print(f"\n{'═'*60}")
    print(f"  NUST Admission Helper is starting...")
    print(f"{'─'*60}")
    print(f"  URL:          {url}")
    print(f"  Ollama:       {'✓ Running' if ollama_status['running'] else '✗ Offline (start with: ollama serve)'}")
    print(f"  AI Model:     {os.getenv('MODEL', MODEL_NAME)}")
    print(f"  Knowledge DB: {'✓ Ready' if rag_ready else '⚠️  Not ready (check dependencies)'}")
    print(f"{'─'*60}")
    print(f"  Opening browser at {url} ...")
    print(f"  Press Ctrl+C to stop the server")
    print(f"{'═'*60}\n")


def main():
    """Main entry point."""
    print_banner()
    create_directories()
    check_ram()

    print(f"\n{'─'*60}")
    print("  System Checks")
    print(f"{'─'*60}")

    # Check index.html
    html_exists = check_index_html()

    # Check Ollama
    ollama_status = check_ollama()

    # Load knowledge base
    rag_ready = load_static_knowledge()

    # Print startup summary
    print_startup_summary(ollama_status, rag_ready)

    # Schedule browser open
    server_url = f"http://{HOST}:{PORT}"
    open_browser_delayed(server_url, delay=2.0)

    # Start FastAPI server
    try:
        import uvicorn
        uvicorn.run(
            "ui.app:app",
            host=HOST,
            port=PORT,
            reload=False,
            log_level="info",
            access_log=True,
        )
    except ImportError:
        print("\n✗ uvicorn is not installed.")
        print("  Run: pip install uvicorn[standard]")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nServer stopped by user. Goodbye!")


if __name__ == "__main__":
    main()
