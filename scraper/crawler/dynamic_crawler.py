import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import hashlib

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scrap import scrap_3
from firebase import firebase_write

MAX_LINKS = 30


def make_document_id(url: str) -> str:
    """Stable doc ID so repeated runs map the same URL to one Firestore doc."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
    return f"policy_{digest}"


def get_links(url: str, starter_url: str) -> list:
    """Fetch a page and return up to MAX_LINKS same-subdomain hrefs."""
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


def crawler(starter_url: str, max_pages: int = 10):
    """
    DFS crawler:
      1. From each page, gather the first 30 hyperlinks
      2. Scrape & score each page
      3. Pages with score > 0  → save to Firebase + local JSON
         Pages with all scores negative → print "nothing relevant" message
    """
    stack   = [starter_url]
    visited = set()
    all_data = []
    pages_crawled = 0
    uploaded_count = 0
    duplicate_count = 0

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
            # ── Scrape & score ──────────────────────────────────
            page_data = scrap_3.scrape_policy_page_final(url)
            page_data["document_id"] = make_document_id(page_data["url"])
            all_data.append(page_data)

            score = page_data["score"]
            label = "✅ RELEVANT" if score > 0 else "❌ not relevant"
            print(f"   Score: {score}  {label}  — {page_data['title']}")

            # ── Save relevant pages to Firebase immediately ──────
            if page_data["relevant"]:
                doc_id = page_data["document_id"]
                page_url = page_data["url"]

                if doc_id in existing_doc_ids or page_url in existing_urls:
                    duplicate_count += 1
                    print(f"   Skipping duplicate (already in Firebase): {page_data['title']}")
                else:
                    wrote = firebase_write.upload_scraped_policy(page_data, skip_if_exists=True)
                    if wrote:
                        uploaded_count += 1
                        existing_doc_ids.add(doc_id)
                        existing_urls.add(page_url)

            # ── Gather links → push unvisited onto stack ─────────
            links = get_links(url, starter_url)
            for link in reversed(links):
                if link not in visited:
                    stack.append(link)

            time.sleep(0.5)

        except Exception as e:
            print(f"[crawler] Error on {url}: {e}")
            continue

    # ── Filter & output ────────────────────────────────────────
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

    # ── Save ALL scored data to local JSON as well ─────────────
    output = {
        "summary": {
            "total_crawled":    len(all_data),
            "relevant_count":   len(relevant),
            "irrelevant_count": len(irrelevant),
            "firebase_uploaded": uploaded_count,
            "firebase_duplicates_skipped": duplicate_count,
        },
        "relevant_pages":   sorted(relevant,   key=lambda x: x["score"], reverse=True),
        "irrelevant_pages": sorted(irrelevant, key=lambda x: x["score"], reverse=True)
    }

    with open("policies.json", "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n💾 Saved to policies.json  ({len(all_data)} total pages)")
    print(
        f"🔥 Firebase policies uploaded: {uploaded_count} | duplicates skipped: {duplicate_count}\n"
    )


# ── Entry point ────────────────────────────────────────────────
starter_url = "https://catalog.purdue.edu/content.php?catoid=15&navoid=18634"
crawler(starter_url, max_pages=50)