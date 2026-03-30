"""
NUST Website Scraper
Scrapes NUST admission-related pages and saves raw data to data/raw/
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
import sys

import requests
from bs4 import BeautifulSoup

# Ensure project root is in path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "data" / "scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("NUSTScraper")

# Directories
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SCRAPE_TARGETS = [
    # ── Primary: ugadmissions.nust.edu.pk (official UG admissions portal) ──
    {
        "url": "https://ugadmissions.nust.edu.pk/",
        "category": "admissions",
        "description": "UG Admissions home — NET schedule, overview",
    },
    {
        "url": "https://ugadmissions.nust.edu.pk/broadfeatures",
        "category": "net",
        "description": "Broad features of NET exam",
    },
    {
        "url": "https://ugadmissions.nust.edu.pk/eligibilitycriteria",
        "category": "eligibility",
        "description": "Eligibility criteria for all programmes",
    },
    {
        "url": "https://ugadmissions.nust.edu.pk/howtoapply",
        "category": "admissions",
        "description": "How to apply — step by step process",
    },
    {
        "url": "https://ugadmissions.nust.edu.pk/faqs",
        "category": "general",
        "description": "Frequently asked questions",
    },
    # ── Secondary: nust.edu.pk (fees, scholarships, general info) ──
    {
        "url": "https://nust.edu.pk/admissions/fee-structure/",
        "category": "fees",
        "description": "Fee structure page",
    },
    {
        "url": "https://nust.edu.pk/admissions/scholarships/",
        "category": "scholarships",
        "description": "Scholarships information",
    },
    {
        "url": "https://nust.edu.pk/admissions/important-dates/",
        "category": "dates",
        "description": "Important admission dates",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


class NUSTScraper:
    """Scraper for NUST admission-related pages."""

    def __init__(self):
        self.pages_scraped = 0
        self.last_refresh: str = ""
        self.ready = False
        self._errors: list[str] = []
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _fetch_page(self, url: str, timeout: int = 15) -> str | None:
        """Fetch a single page and return its HTML content."""
        try:
            response = self.session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return response.text
        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching %s", url)
            self._errors.append(f"Timeout: {url}")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error fetching %s", url)
            self._errors.append(f"ConnectionError: {url}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning("HTTP error %s for %s", e.response.status_code, url)
            self._errors.append(f"HTTP {e.response.status_code}: {url}")
            return None
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", url, str(e))
            self._errors.append(f"Error: {url} - {str(e)}")
            return None

    def _extract_text(self, html: str, url: str, category: str) -> dict:
        """Extract meaningful text from HTML using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript"]):
            tag.decompose()

        # Extract title
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)

        # Extract main content - try various content selectors
        main_content = None
        for selector in ["main", "article", ".content", "#content", ".entry-content", ".page-content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.find("body") or soup

        # Extract all text paragraphs and headings
        text_blocks = []
        for tag in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th"]):
            text = tag.get_text(separator=" ", strip=True)
            if text and len(text) > 20:  # Filter out very short snippets
                text_blocks.append(text)

        # Extract tables separately
        tables = []
        for table in main_content.find_all("table"):
            rows = []
            for row in table.find_all("tr"):
                cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                tables.append("\n".join(rows))

        # Extract links for context
        links = []
        for a_tag in main_content.find_all("a", href=True):
            link_text = a_tag.get_text(strip=True)
            href = a_tag["href"]
            if link_text and len(link_text) > 3:
                links.append({"text": link_text, "href": href})

        full_text = "\n\n".join(text_blocks)

        return {
            "url": url,
            "category": category,
            "title": title,
            "text": full_text,
            "tables": tables,
            "links": links[:50],  # Limit links
            "scraped_at": datetime.now().isoformat(),
            "char_count": len(full_text),
        }

    def _save_raw(self, data: dict, category: str) -> Path:
        """Save extracted data to data/raw/ with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RAW_DIR / f"{category}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved raw data: %s (%d chars)", filename.name, data.get("char_count", 0))
        return filename

    def scrape_page(self, target: dict) -> bool:
        """Scrape a single page target. Returns True on success."""
        url = target["url"]
        category = target["category"]
        logger.info("Scraping: %s (%s)", url, category)

        html = self._fetch_page(url)
        if html is None:
            logger.warning("Skipping %s due to fetch failure", url)
            return False

        data = self._extract_text(html, url, category)

        if data["char_count"] < 100:
            logger.warning("Very little content extracted from %s (%d chars)", url, data["char_count"])

        self._save_raw(data, category)
        return True

    def scrape_all(self, delay: float = 1.5) -> dict:
        """
        Scrape all configured NUST pages.
        Returns summary dict with pages scraped and any errors.
        """
        logger.info("Starting NUST website scrape - %d targets", len(SCRAPE_TARGETS))
        self.pages_scraped = 0
        self._errors = []
        start_time = time.time()

        for target in SCRAPE_TARGETS:
            success = self.scrape_page(target)
            if success:
                self.pages_scraped += 1
            time.sleep(delay)  # Be polite to the server

        elapsed = time.time() - start_time
        self.last_refresh = datetime.now().isoformat()
        self.ready = self.pages_scraped > 0

        summary = {
            "pages_scraped": self.pages_scraped,
            "total_targets": len(SCRAPE_TARGETS),
            "errors": self._errors,
            "elapsed_seconds": round(elapsed, 2),
            "last_refresh": self.last_refresh,
            "ready": self.ready,
        }

        logger.info(
            "Scrape complete: %d/%d pages in %.2fs",
            self.pages_scraped,
            len(SCRAPE_TARGETS),
            elapsed,
        )
        return summary

    def get_scraper_status(self) -> dict:
        """Return current scraper status."""
        # Check if any raw files exist
        raw_files = list(RAW_DIR.glob("*.json"))
        if raw_files and not self.ready:
            # Files exist from a previous run
            latest = max(raw_files, key=lambda f: f.stat().st_mtime)
            mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
            self.last_refresh = mod_time.isoformat()
            self.pages_scraped = len(raw_files)
            self.ready = True

        return {
            "ready": self.ready,
            "pages_scraped": self.pages_scraped,
            "last_refresh": self.last_refresh or "Never",
        }


# Singleton instance for import
scraper = NUSTScraper()


if __name__ == "__main__":
    s = NUSTScraper()
    result = s.scrape_all()
    print(json.dumps(result, indent=2))
