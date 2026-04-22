import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from pathlib import Path
from typing import Dict, Set, Tuple


# Firebase service-account JSON lives in the BoilerCheck repo root (parents[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIREBASE_KEY_PATH = _REPO_ROOT / "gdg-web-scraping-data-firebase-adminsdk-fbsvc-d6a5997024.json"


def _initialize_firestore_client():
    if not _FIREBASE_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Firebase service-account key not found at {_FIREBASE_KEY_PATH}. "
            "Make sure it lives in the BoilerCheck repo root."
        )
    cred = credentials.Certificate(str(_FIREBASE_KEY_PATH))
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return firestore.client()


db = _initialize_firestore_client()


def fetch_existing_policies() -> Tuple[Set[str], Set[str]]:
    """Return (doc_ids, urls) for every document in the `policies` collection."""
    doc_ids: Set[str] = set()
    urls: Set[str] = set()

    for doc in db.collection("policies").stream():
        data = doc.to_dict() or {}
        doc_ids.add(doc.id)
        url = data.get("url")
        if isinstance(url, str) and url:
            urls.add(url)

    return doc_ids, urls


def _upload_to_collection(
    collection: str, scraped_data: Dict, skip_if_exists: bool
) -> bool:
    doc_id = scraped_data.get("document_id")
    if not doc_id:
        raise ValueError("scraped_data must include 'document_id'")

    doc_ref = db.collection(collection).document(doc_id)
    if skip_if_exists and doc_ref.get().exists:
        return False

    payload = dict(scraped_data)
    payload["last_updated"] = firestore.SERVER_TIMESTAMP
    doc_ref.set(payload, merge=True)
    return True


def upload_scraped_policy(scraped_data: Dict, skip_if_exists: bool = True) -> bool:
    """Upload to `policies`. Returns True if written, False if skipped."""
    wrote = _upload_to_collection("policies", scraped_data, skip_if_exists)
    if wrote:
        print(f"Successfully uploaded/updated: {scraped_data['document_id']}")
    return wrote


def upload_scraped_policy_with_images(scraped_data: Dict, skip_if_exists: bool = True) -> bool:
    """Upload to `policies_with_images`. Returns True if written, False if skipped."""
    wrote = _upload_to_collection("policies_with_images", scraped_data, skip_if_exists)
    if wrote:
        print(f"Successfully uploaded with images: {scraped_data['document_id']}")
    return wrote
