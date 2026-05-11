import json
import time
import logging
import re
from pathlib import Path
from typing import Any
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
BASE_URL = "https://www.shl.com"
OUTPUT_PATH = Path("data/shl_catalog.json")
DELAY = 1.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

ICON_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session

def parse_listing(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    products = []

    rows = soup.select("table.custom-table tbody tr") or soup.select("[data-course-id]")
    for row in rows:
        try:
            link_tag = row.select_one("a[href]")
            if not link_tag: continue
            
            href = link_tag["href"]
            url = href if href.startswith("http") else BASE_URL + href
            
            # Map category icons
            icons = row.select("td span[class*='catalogue__circle']")
            keys = [ICON_MAP[s.get_text(strip=True).upper()] for s in icons if s.get_text(strip=True).upper() in ICON_MAP]

            products.append({
                "name": link_tag.get_text(strip=True),
                "link": url,
                "keys": keys,
                "description": "",
                "job_levels": [],
                "languages": [],
                "duration": "",
                "remote": "no",
                "adaptive": "no",
            })
        except Exception as e:
            logger.debug(f"Row parse error: {e}")
            continue
    return products

def scrape_detail(session: requests.Session, product: dict[str, Any]) -> dict[str, Any]:
    try:
        resp = session.get(product["link"], timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract description
        desc_tag = soup.select_one(".product-catalogue-training-calendar__overview--content p")
        if not desc_tag:
            desc_tag = soup.select_one("meta[name='description']")
            product["description"] = desc_tag["content"].strip() if desc_tag and desc_tag.has_attr("content") else ""
        else:
            product["description"] = desc_tag.get_text(strip=True)

        # Extract metadata
        for row in soup.select(".product-catalogue__key-details li, .product-detail__item"):
            label = row.select_one("dt, .product-detail__label")
            value = row.select_one("dd, .product-detail__value")
            if not label or not value: continue
            
            key = label.get_text(strip=True).lower()
            val = value.get_text(separator=", ", strip=True)
            if "job level" in key:
                product["job_levels"] = [s.strip() for s in val.split(",") if s.strip()]
            elif "language" in key:
                product["languages"] = [s.strip() for s in val.split(",") if s.strip()]
            elif "duration" in key or "time" in key:
                product["duration"] = val

        # Handle badges
        for badge in soup.select(".product-catalogue__tag"):
            text = badge.get_text(strip=True).lower()
            if "remote" in text: product["remote"] = "yes"
            if "adaptive" in text: product["adaptive"] = "yes"

    except Exception as e:
        logger.warning(f"Detail scrape failed for {product['name']}: {e}")
    return product

def run_scraper(detail: bool = True):
    session = get_session()
    all_products = []
    seen = set()

    # Pagination logic
    start = 0
    while True:
        url = f"{CATALOG_URL}?type=1&start={start}"
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            
            page_products = parse_listing(resp.text)
            if not page_products: break
            
            for p in page_products:
                if p["link"] not in seen:
                    seen.add(p["link"])
                    all_products.append(p)
            
            logger.info(f"Collected {len(all_products)} product stubs...")
            start += 12
            time.sleep(DELAY)
            if start > 600: break # Safety break
        except Exception as e:
            logger.error(f"Listing fetch failed at start={start}: {e}")
            break

    if detail:
        logger.info(f"Fetching details for {len(all_products)} products...")
        for i, p in enumerate(all_products):
            all_products[i] = scrape_detail(session, p)
            if (i+1) % 10 == 0: logger.info(f"Progress: {i+1}/{len(all_products)}")
            time.sleep(DELAY)

    return all_products

def save(products: list[dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(products)} records to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-detail", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    import argparse
    args = parser.parse_args()

    results = run_scraper(detail=not args.no_detail)
    save(results, Path(args.output))
