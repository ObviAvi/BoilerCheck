"""
One-off cleanup: remove sections whose `text` has fewer than MIN_WORDS words
from every document in both the `policies` and `policies_with_images`
Firestore collections.

Run from this folder (or anywhere — the script adds its own dir to sys.path):
    python clean_short_sections.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from firebase_write import db

COLLECTIONS = ["policies", "policies_with_images"]
MIN_WORDS = 10


def _word_count(text) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def _clean_collection(collection: str) -> None:
    scanned = 0
    updated = 0
    removed = 0

    print(f"\n=== {collection} ===")

    for snap in db.collection(collection).stream():
        scanned += 1
        data = snap.to_dict() or {}
        sections = data.get("sections")
        if not isinstance(sections, list) or not sections:
            continue

        kept = [
            sec for sec in sections
            if isinstance(sec, dict) and _word_count(sec.get("text")) >= MIN_WORDS
        ]
        dropped = len(sections) - len(kept)
        if dropped == 0:
            continue

        db.collection(collection).document(snap.id).update({"sections": kept})
        updated += 1
        removed += dropped
        print(f"  {snap.id}: removed {dropped} short section(s) ({len(kept)} kept)")

    print(
        f"Scanned {scanned} doc(s), updated {updated} doc(s), "
        f"removed {removed} section(s) with <{MIN_WORDS} words."
    )


def main() -> None:
    for collection in COLLECTIONS:
        _clean_collection(collection)


if __name__ == "__main__":
    main()
