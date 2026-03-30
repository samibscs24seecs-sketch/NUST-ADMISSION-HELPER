"""
NUST Admission Helper - Scheduled Update Job
Checks for updates every 30 days using hash comparison.
Sends desktop notifications via plyer.
"""

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Setup file-based logging
LOG_FILE = ROOT / "data" / "scheduler.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE), encoding="utf-8"),
    ],
)
logger = logging.getLogger("UpdateScheduler")

UPDATE_INTERVAL_DAYS = 30
HASH_FILE = ROOT / "data" / "content_hash.json"
DATA_DIR = ROOT / "data" / "raw"


def _compute_directory_hash(directory: Path) -> str:
    """
    Compute a SHA256 hash of all JSON files in a directory.
    Used for detecting content changes between scrapes.
    """
    hasher = hashlib.sha256()
    json_files = sorted(directory.glob("*.json"))

    if not json_files:
        return ""

    for fpath in json_files:
        try:
            with open(fpath, encoding="utf-8") as f:
                # Hash just the content, not timestamps (which change every scrape)
                data = json.load(f)
                text = data.get("text", "") + data.get("title", "")
                hasher.update(text.encode("utf-8", errors="replace"))
        except Exception:
            hasher.update(fpath.name.encode())

    return hasher.hexdigest()


def _load_saved_hash() -> dict:
    """Load previously saved content hash."""
    if HASH_FILE.exists():
        try:
            with open(HASH_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"hash": "", "saved_at": ""}


def _save_hash(hash_value: str):
    """Save the current content hash."""
    data = {
        "hash": hash_value,
        "saved_at": datetime.now().isoformat(),
    }
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _send_notification(title: str, message: str):
    """Send a Windows desktop notification via plyer."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="NUST Admission Helper",
            timeout=10,
        )
        logger.info("Desktop notification sent: %s", title)
    except ImportError:
        logger.warning("plyer not installed — desktop notifications disabled.")
    except Exception as e:
        logger.warning("Notification failed: %s", e)


def _run_update() -> dict:
    """
    Run the full update pipeline:
    1. Scrape NUST website
    2. Compare content hash with stored hash
    3. If changed: clean data + re-embed into ChromaDB
    4. Notify user
    Returns summary dict.
    """
    logger.info("=" * 60)
    logger.info("Starting scheduled NUST data update")
    logger.info("=" * 60)

    result = {
        "started_at": datetime.now().isoformat(),
        "scrape": {},
        "content_changed": False,
        "embedding_update": {},
        "status": "pending",
    }

    # Step 1: Scrape
    try:
        from scraper.scrape import NUSTScraper
        scraper = NUSTScraper()
        scrape_result = scraper.scrape_all()
        result["scrape"] = scrape_result
        logger.info("Scrape completed: %d pages", scrape_result.get("pages_scraped", 0))
    except Exception as e:
        logger.error("Scrape failed: %s", e)
        result["status"] = "error"
        result["error"] = str(e)
        _send_notification(
            "NUST Update Failed",
            f"Scraping failed: {str(e)[:100]}",
        )
        return result

    # Step 2: Hash comparison
    new_hash = _compute_directory_hash(DATA_DIR)
    saved = _load_saved_hash()
    old_hash = saved.get("hash", "")

    if new_hash == old_hash and old_hash:
        logger.info("Content unchanged (hash match). Skipping embedding update.")
        result["content_changed"] = False
        result["status"] = "no_change"
        _send_notification(
            "NUST Data Up-to-Date",
            "No changes detected on NUST website. Knowledge base is current.",
        )
        return result

    logger.info("Content changed! Old hash: %s... New hash: %s...", old_hash[:16], new_hash[:16])
    result["content_changed"] = True

    # Step 3: Clean and re-embed
    try:
        from scraper.cleaner import NUSTCleaner
        cleaner = NUSTCleaner()
        clean_result = cleaner.clean_all()
        logger.info("Cleaning done: %d chunks", clean_result.get("chunks_created", 0))

        chunks = cleaner.load_cleaned_chunks()
        if chunks:
            from rag.embedder import ChromaDBManager
            db = ChromaDBManager()
            docs_added = db.upsert_documents(chunks)
            result["embedding_update"] = {
                "chunks_processed": len(chunks),
                "documents_upserted": docs_added,
            }
            logger.info("Embedded %d documents into ChromaDB", docs_added)
        else:
            logger.warning("No chunks to embed after cleaning.")
            result["embedding_update"] = {"chunks_processed": 0, "documents_upserted": 0}

    except Exception as e:
        logger.error("Embedding update failed: %s", e)
        result["status"] = "partial_error"
        result["error"] = str(e)
        return result

    # Step 4: Save new hash
    _save_hash(new_hash)

    result["status"] = "success"
    result["completed_at"] = datetime.now().isoformat()

    _send_notification(
        "NUST Knowledge Base Updated!",
        f"Successfully scraped {result['scrape'].get('pages_scraped', 0)} pages "
        f"and updated {result['embedding_update'].get('documents_upserted', 0)} documents.",
    )

    logger.info("Update completed successfully.")
    return result


class UpdateScheduler:
    """
    Background scheduler that checks for NUST website updates every 30 days.
    Uses threading for non-blocking execution.
    """

    def __init__(self, interval_days: int = UPDATE_INTERVAL_DAYS):
        self.interval_days = interval_days
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_run: datetime | None = None
        self._next_run: datetime | None = None

    def _should_run(self) -> bool:
        """Check if enough time has passed since last update."""
        if self._last_run is None:
            # On fresh start, check the saved hash file timestamp instead of
            # scraping immediately — avoids re-contaminating ChromaDB after a wipe.
            saved = _load_saved_hash()
            saved_at = saved.get("saved_at", "")
            if saved_at:
                try:
                    last = datetime.fromisoformat(saved_at)
                    if datetime.now() - last < timedelta(days=self.interval_days):
                        self._last_run = last
                        self._next_run = last + timedelta(days=self.interval_days)
                        return False
                except ValueError:
                    pass
            # No valid saved timestamp — schedule first run after a 1-hour delay
            # so the server fully starts up before the scraper hits the network.
            self._last_run = datetime.now()
            self._next_run = self._last_run + timedelta(hours=1)
            return False
        return datetime.now() >= self._next_run

    def _scheduler_loop(self):
        """Main scheduler loop — runs in background thread."""
        logger.info(
            "Scheduler started. Will check for updates every %d days.",
            self.interval_days,
        )

        while self._running:
            if self._should_run():
                logger.info("Running scheduled update check...")
                try:
                    result = _run_update()
                    self._last_run = datetime.now()
                    self._next_run = self._last_run + timedelta(days=self.interval_days)
                    logger.info(
                        "Next update scheduled for: %s",
                        self._next_run.strftime("%Y-%m-%d %H:%M"),
                    )
                except Exception as e:
                    logger.error("Scheduler loop error: %s", e)
                    # Retry in 24 hours on failure
                    self._next_run = datetime.now() + timedelta(hours=24)

            # Sleep for 1 hour between checks
            for _ in range(3600):
                if not self._running:
                    break
                time.sleep(1)

    def start(self):
        """Start the background scheduler thread."""
        if self._running:
            logger.warning("Scheduler already running.")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True,
            name="NUSTUpdateScheduler",
        )
        self._thread.start()
        logger.info("Scheduler thread started (daemon=True).")

    def stop(self):
        """Stop the scheduler gracefully."""
        logger.info("Stopping scheduler...")
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped.")

    def run_update_now(self) -> dict:
        """
        Trigger an immediate update (blocking).
        Returns the update result dict.
        """
        logger.info("Manual update triggered.")
        result = _run_update()
        self._last_run = datetime.now()
        self._next_run = self._last_run + timedelta(days=self.interval_days)
        return result

    def get_status(self) -> dict:
        """Return current scheduler status."""
        return {
            "running": self._running,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "next_run": self._next_run.isoformat() if self._next_run else None,
            "interval_days": self.interval_days,
        }


# Singleton
scheduler = UpdateScheduler()


if __name__ == "__main__":
    print("Running manual update now...")
    result = _run_update()
    print(json.dumps(result, indent=2))
