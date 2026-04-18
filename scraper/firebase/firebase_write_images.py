import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os
from typing import Dict, Set, Tuple


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _initialize_firestore_client():
    key_path = os.path.join(
        _project_root(), "gdg-web-scraping-data-firebase-adminsdk-fbsvc-d515cca4af.json"
    )
    cred = credentials.Certificate(key_path)

    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

    return firestore.client()


db = _initialize_firestore_client()


def fetch_existing_policies() -> Tuple[Set[str], Set[str]]:
    """Return existing policy doc IDs and URLs from Firestore."""
    doc_ids: Set[str] = set()
    urls: Set[str] = set()

    for doc in db.collection("policies").stream():
        data = doc.to_dict() or {}
        doc_ids.add(doc.id)
        url = data.get("url")
        if isinstance(url, str) and url:
            urls.add(url)

    return doc_ids, urls


def upload_scraped_policy(scraped_data: Dict, skip_if_exists: bool = True) -> bool:
    """Upload policy document. Returns True if written, False if skipped."""
    doc_id = scraped_data.get("document_id")
    if not doc_id:
        raise ValueError("scraped_data must include 'document_id'")

    doc_ref = db.collection("policies").document(doc_id)
    if skip_if_exists and doc_ref.get().exists:
        return False

    payload = dict(scraped_data)
    payload["last_updated"] = firestore.SERVER_TIMESTAMP

    doc_ref.set(payload, merge=True)
    print(f"Successfully uploaded/updated: {doc_id}")
    return True

"uploads the scraped policy with image description to firebase to a new collection"
"called policies_with_images"

def upload_scraped_policy_with_images(scraped_data: Dict, skip_if_exists: bool = True) -> bool:
    """Upload to policies_with_images collection. Returns True if written, False if skipped."""
    doc_id = scraped_data.get("document_id")
    if not doc_id:
        raise ValueError("scraped_data must include 'document_id'")

    doc_ref = db.collection("policies_with_images").document(doc_id)
    if skip_if_exists and doc_ref.get().exists:
        return False

    payload = dict(scraped_data)
    payload["last_updated"] = firestore.SERVER_TIMESTAMP

    doc_ref.set(payload, merge=True)
    print(f"Successfully uploaded to policies_with_images: {doc_id}")
    return True