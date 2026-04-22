"""
Scrape a single web page (text only) and upload it to the `policies` Firestore
collection.

Sections are split by h2/h3 headers in scrape_3.scrape_policy_page_final and
stored as-is; finer-grained chunking happens later in
backend/ingest_policies_no_images.py via RecursiveCharacterTextSplitter.

Usage:
    python scrape_single.py <url>
    python scrape_single.py <url> --force        # overwrite existing doc
    python scrape_single.py <url> --min-score 0  # skip the relevance gate
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from scrape.scrape_3 import SCORE_THRESHOLD, scrape_policy_page_final
from firebase import firebase_write


def make_document_id(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
    return f"policy_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape one page into `policies`.")
    parser.add_argument("url", help="Full page URL to scrape.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the Firestore doc if it already exists.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=SCORE_THRESHOLD,
        help=f"Skip upload if page score < this value (default: {SCORE_THRESHOLD}).",
    )
    args = parser.parse_args()

    print(f"Scraping: {args.url}")
    page_data = scrape_policy_page_final(args.url)
    page_data["document_id"] = make_document_id(page_data["url"])
    page_data.pop("images", None)

    score = page_data.get("score", 0)
    sections = page_data.get("sections") or []
    print(f"  Title:    {page_data.get('title', 'Unknown')}")
    print(f"  Sections: {len(sections)}")
    print(f"  Score:    {score}  (threshold: {args.min_score})")

    if score < args.min_score:
        print(f"  Skipped: score {score} < {args.min_score}. Re-run with --min-score 0 to force.")
        return

    wrote = firebase_write.upload_scraped_policy(
        page_data, skip_if_exists=not args.force
    )
    if wrote:
        print(f"  Uploaded to `policies`: {page_data['document_id']}")
    else:
        print(
            f"  Already exists in `policies` ({page_data['document_id']}). "
            "Re-run with --force to overwrite."
        )


if __name__ == "__main__":
    main()
