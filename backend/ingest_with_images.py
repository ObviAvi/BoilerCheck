"""
Reads Firestore collection `policies_with_images`, creates one vector entry
per text section and one vector entry per image description, embeds with
all-MiniLM-L6-v2, and upserts into Pinecone.

Run whenever the source data changes:
    python ingest_with_images.py
"""

import os
from pathlib import Path
from importlib import import_module

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# .env and the Firebase service-account JSON live in the BoilerCheck repo root (parents[1]).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIREBASE_KEY_PATH = _REPO_ROOT / "gdg-web-scraping-data-firebase-adminsdk-fbsvc-d6a5997024.json"

load_dotenv(_REPO_ROOT / ".env")

COLLECTION_NAME = os.getenv("POLICIES_COLLECTION", "policies_with_images")


def _initialize_firestore_client():
    if not _FIREBASE_KEY_PATH.exists():
        raise RuntimeError(
            f"Firebase service-account key not found at {_FIREBASE_KEY_PATH}. "
            "Make sure it lives in the BoilerCheck repo root."
        )

    try:
        firebase_admin = import_module("firebase_admin")
        firebase_credentials = import_module("firebase_admin.credentials")
        firebase_firestore = import_module("firebase_admin.firestore")
    except ImportError as import_error:
        raise RuntimeError(
            "firebase-admin is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from import_error

    if not firebase_admin._apps:
        cred = firebase_credentials.Certificate(str(_FIREBASE_KEY_PATH))
        firebase_admin.initialize_app(cred)

    print(f"Using Firebase service-account key: {_FIREBASE_KEY_PATH.name}")
    return firebase_firestore.client()


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _base_meta(record: dict) -> dict:
    doc_id = _safe_str(record.get("document_id")) or _safe_str(record.get("id"))
    title = _safe_str(record.get("title")) or doc_id or "Untitled Source"
    images = record.get("images") if isinstance(record.get("images"), list) else []
    first_image_url = ""
    for image in images:
        if not isinstance(image, dict):
            continue
        first_image_url = _safe_str(image.get("source_url"))
        if first_image_url:
            break

    return {
        "document_id": doc_id,
        "title": title,
        "domain": _safe_str(record.get("domain")),
        "url": _safe_str(record.get("url")) or first_image_url,
        "effective_date": _safe_str(record.get("effective_date")),
        "has_structure": bool(record.get("has_structure", False)),
        "last_revised": _safe_str(record.get("last_revised")),
    }


def _build_text_chunks(record: dict, base_meta: dict) -> list[Document]:
    chunks: list[Document] = []
    sections = record.get("sections") if isinstance(record.get("sections"), list) else []

    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        text = _safe_str(section.get("text"))
        if not text:
            continue

        section_title = _safe_str(section.get("section_title")) or f"Section {idx + 1}"
        chunks.append(
            Document(
                page_content=text,
                metadata={
                    **base_meta,
                    "chunk_type": "text",
                    "section_title": section_title,
                    "subsection_title": "",
                    "source_key": section_title,
                },
            )
        )

    return chunks


def _build_image_chunks(record: dict, base_meta: dict) -> list[Document]:
    chunks: list[Document] = []
    images = record.get("images") if isinstance(record.get("images"), list) else []

    for idx, image in enumerate(images):
        if not isinstance(image, dict):
            continue

        description = _safe_str(image.get("description"))
        source_url = _safe_str(image.get("source_url"))
        if not description:
            continue

        filename = _safe_str(image.get("filename"))
        image_label = filename or f"image_{idx + 1}"
        chunks.append(
            Document(
                page_content=description,
                metadata={
                    **base_meta,
                    "chunk_type": "image",
                    "section_title": "Image",
                    "subsection_title": image_label,
                    "source_key": f"Image/{image_label}",
                    "image_description": description,
                    "image_source_url": source_url,
                    "image_filename": filename,
                    "image_format": _safe_str(image.get("format")),
                    "image_type": _safe_str(image.get("image_type")),
                    "image_md5": _safe_str(image.get("md5")),
                    "image_width": _safe_int(image.get("width", 0)),
                    "image_height": _safe_int(image.get("height", 0)),
                    "image_public_url": _safe_str(image.get("public_url")),
                },
            )
        )

    return chunks


def load_chunks_from_firestore() -> list[Document]:
    """
    Build one vector entry per text section and one per image description from
    Firestore records in the configured collection.
    """
    db = _initialize_firestore_client()
    chunks: list[Document] = []

    for snap in db.collection(COLLECTION_NAME).stream():
        record = snap.to_dict() or {}
        if not isinstance(record, dict):
            continue

        if not record.get("document_id"):
            record["document_id"] = snap.id

        base_meta = _base_meta(record)
        chunks.extend(_build_text_chunks(record, base_meta))
        chunks.extend(_build_image_chunks(record, base_meta))

    return chunks


def main():
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not set in .env")

    print(f"Loading data from Firestore collection '{COLLECTION_NAME}' ...")
    chunks = load_chunks_from_firestore()
    text_count = sum(1 for c in chunks if (c.metadata or {}).get("chunk_type") == "text")
    image_count = sum(1 for c in chunks if (c.metadata or {}).get("chunk_type") == "image")
    print(f"  {len(chunks)} total chunks ready to upsert")
    print(f"  text chunks: {text_count}")
    print(f"  image chunks: {image_count}")

    if not chunks:
        raise RuntimeError("No chunks were produced from Firestore data.")

    print("Loading embedding model ...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Upserting into Pinecone index '{index_name}' ...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        text_key="text",
    )

    print(f"Embedded image descriptions: {image_count}")
    print(f"Done — {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    main()
