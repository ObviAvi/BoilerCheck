"""
One-off cleanup: remove image records whose `image_type == "error"` from every
document in the `policies_with_images` Firestore collection.

Run from this folder (or anywhere — the script adds its own dir to sys.path):
    python clean_error_images.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from firebase_write import db

COLLECTION = "policies_with_images"
BAD_TYPE = "error"


def main() -> None:
    scanned = 0
    updated = 0
    removed = 0

    for snap in db.collection(COLLECTION).stream():
        scanned += 1
        data = snap.to_dict() or {}
        images = data.get("images")
        if not isinstance(images, list) or not images:
            continue

        kept = [
            img for img in images
            if not (isinstance(img, dict) and img.get("image_type") == BAD_TYPE)
        ]
        dropped = len(images) - len(kept)
        if dropped == 0:
            continue

        db.collection(COLLECTION).document(snap.id).update({"images": kept})
        updated += 1
        removed += dropped
        print(f"  {snap.id}: removed {dropped} error image(s) ({len(kept)} kept)")

    print(
        f"\nScanned {scanned} doc(s), updated {updated} doc(s), "
        f"removed {removed} image record(s) with image_type='{BAD_TYPE}'."
    )


if __name__ == "__main__":
    main()
