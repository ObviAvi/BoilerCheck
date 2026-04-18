import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import hashlib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scrap import scrap_3
from firebase import firebase_write
from crawler import classify_images
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

MAX_LINKS = 30

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "crawler", "data", "images")

MIN_IMAGE_WIDTH  = 50
MIN_IMAGE_HEIGHT = 50

IMAGE_ATTRS = [
    ("img", "src"),
    ("img", "data-src"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_document_id(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
    return f"policy_{digest}"


def get_links(url: str, starter_url: str) -> list:
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []

        soup = BeautifulSoup(r.content, 'html.parser')
        links = []

        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(url, href).split('#')[0]
            parsed = urlparse(full_url)

            if parsed.netloc != urlparse(starter_url).netloc:
                continue
            if not parsed.path.startswith(urlparse(starter_url).path):
                continue
            if full_url not in links:
                links.append(full_url)
            if len(links) >= MAX_LINKS:
                break

        return links

    except Exception as e:
        print(f"[get_links] Error fetching {url}: {e}")
        return []


# ── Image helpers ─────────────────────────────────────────────────────────────

def resolve_url(src, page_url):
    if not src or src.startswith("data:"):
        return None
    return urljoin(page_url, src.strip())


def download_image(img_url):
    try:
        resp = requests.get(img_url, timeout=10)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"  Warning: could not download {img_url}: {e}")
        return None


def get_image_dimensions(img_bytes):
    if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        w = int.from_bytes(img_bytes[16:20], "big")
        h = int.from_bytes(img_bytes[20:24], "big")
        return w, h
    if img_bytes[:2] == b'\xff\xd8':
        i = 2
        while i < len(img_bytes) - 8:
            if img_bytes[i] != 0xFF:
                break
            marker = img_bytes[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h = int.from_bytes(img_bytes[i + 5:i + 7], "big")
                w = int.from_bytes(img_bytes[i + 7:i + 9], "big")
                return w, h
            length = int.from_bytes(img_bytes[i + 2:i + 4], "big")
            i += 2 + length
    return 0, 0


def guess_extension(img_bytes, fallback_url):
    sigs = {b'\x89PNG': "png", b'\xff\xd8': "jpg",
            b'GIF8': "gif", b'RIFF': "webp"}
    for sig, ext in sigs.items():
        if img_bytes[:len(sig)] == sig:
            return ext
    path = urlparse(fallback_url).path
    if "." in path:
        return path.rsplit(".", 1)[-1].lower()[:5]
    return "bin"


def extract_images(soup, page_url, document_id, seen_hashes):
    """Extract images from page and save to disk. Classification handled separately."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    image_records = []

    candidate_urls = []
    for tag, attr in IMAGE_ATTRS:
        for el in soup.find_all(tag):
            raw = el.get(attr, "")
            abs_url = resolve_url(raw, page_url)
            if abs_url:
                candidate_urls.append(abs_url)

    for img_url in candidate_urls:
        img_bytes = download_image(img_url)
        if not img_bytes:
            continue

        width, height = get_image_dimensions(img_bytes)
        if 0 < width < MIN_IMAGE_WIDTH or 0 < height < MIN_IMAGE_HEIGHT:
            continue

        img_hash = hashlib.md5(img_bytes).hexdigest()
        if img_hash in seen_hashes:
            continue
        seen_hashes.add(img_hash)

        ext      = guess_extension(img_bytes, img_url)
        filename = f"{document_id}_{img_hash[:8]}.{ext}"
        filepath = os.path.join(IMAGE_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        print(f"  Saved image: {filename} ({width}x{height})")

        image_records.append({
            "filename":    filename,
            "source_url":  img_url,
            "width":       width,
            "height":      height,
            "format":      ext,
            "md5":         img_hash,
            "description": "",
            "image_type":  "",
            "public_url":  "",
        })

    return image_records


# ── Crawler ───────────────────────────────────────────────────────────────────

def crawler(starter_url: str, max_pages: int = 10):
    stack    = [starter_url]
    visited  = set()
    all_data = []
    pages_crawled  = 0
    uploaded_count = 0
    duplicate_count = 0
    seen_hashes = set()

    existing_doc_ids, existing_urls = firebase_write.fetch_existing_policies()
    print(
        f"[Firebase] Found {len(existing_doc_ids)} existing policy docs; duplicates will be skipped."
    )

    while stack and pages_crawled < max_pages:
        url = stack.pop()

        if url in visited:
            continue

        print(f"[DFS] Crawling ({pages_crawled + 1}/{max_pages}): {url}")
        visited.add(url)
        pages_crawled += 1

        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                print(f"  Skipping (status {r.status_code})")
                continue

            soup = BeautifulSoup(r.content, 'html.parser')

            # ── Scrape text + score ──────────────────────────────
            page_data = scrap_3.scrape_policy_page_final(url)
            page_data["document_id"] = make_document_id(page_data["url"])

            # ── Extract images (raw, no classification yet) ──────
            page_data["images"] = extract_images(
                soup, url, page_data["document_id"], seen_hashes
            )

            # ── Classify images via classify_images.py ───────────
            classify_images.classify_images_for_data([page_data], None)

            all_data.append(page_data)

            score = page_data["score"]
            label = "✅ RELEVANT" if score > 0 else "❌ not relevant"
            print(f"   Score: {score}  {label}  — {page_data['title']}")
            print(f"   Images: {len(page_data['images'])}")

            # ── Upload to policies (no images) if new ────────────
            doc_id   = page_data["document_id"]
            page_url = page_data["url"]

            if doc_id in existing_doc_ids or page_url in existing_urls:
                duplicate_count += 1
                print(f"   Skipping duplicate in policies: {page_data['title']}")
            else:
                page_data_no_images = {k: v for k, v in page_data.items() if k != "images"}
                wrote = firebase_write.upload_scraped_policy(page_data_no_images, skip_if_exists=True)
                if wrote:
                    uploaded_count += 1
                    existing_doc_ids.add(doc_id)
                    existing_urls.add(page_url)

            # ── Always upload to policies_with_images ────────────
            firebase_write.upload_scraped_policy_with_images(page_data, skip_if_exists=False)

            # ── Discover links ───────────────────────────────────
            links = get_links(url, starter_url)
            for link in reversed(links):
                if link not in visited:
                    stack.append(link)

            time.sleep(0.5)

        except Exception as e:
            print(f"[crawler] Error on {url}: {e}")
            continue

    # ── Filter & output ───────────────────────────────────────────────────────
    relevant   = [d for d in all_data if d["score"] > 0]
    irrelevant = [d for d in all_data if d["score"] <= 0]

    print("\n" + "=" * 60)

    if not relevant:
        print("⚠️  Nothing relevant to Purdue Policy was found.")
        print(f"   ({len(irrelevant)} pages crawled, all scored negative)\n")
    else:
        relevant_sorted = sorted(relevant, key=lambda x: x["score"], reverse=True)
        print(f"✅ Found {len(relevant_sorted)} Purdue Policy-relevant page(s):\n")
        for d in relevant_sorted:
            print(f"   [{d['score']:>4}]  {d['title']}")
            print(f"          {d['url']}")

    output = {
        "summary": {
            "total_crawled":              len(all_data),
            "relevant_count":             len(relevant),
            "irrelevant_count":           len(irrelevant),
            "firebase_uploaded":          uploaded_count,
            "firebase_duplicates_skipped": duplicate_count,
        },
        "relevant_pages":   sorted(relevant,   key=lambda x: x["score"], reverse=True),
        "irrelevant_pages": sorted(irrelevant, key=lambda x: x["score"], reverse=True),
    }

    with open("policies.json", "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n💾 Saved to policies.json  ({len(all_data)} total pages)")
    print(
        f"🔥 Firebase policies uploaded: {uploaded_count} | duplicates skipped: {duplicate_count}\n"
    )


# ── Entry point ───────────────────────────────────────────────────────────────
starter_url = "https://catalog.purdue.edu/content.php?catoid=15&navoid=18634"
crawler(starter_url, max_pages=50)