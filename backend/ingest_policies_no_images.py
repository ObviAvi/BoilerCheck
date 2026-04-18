"""
Ingestion script for BoilerCheck policies (text only).

Reads Firestore collection `policies`, creates one vector entry per text
section, embeds with all-MiniLM-L6-v2, and upserts into Pinecone.

This script intentionally ignores images and focuses on the text-only schema.

Run:
    python ingest_policies_no_images.py
"""

from __future__ import annotations

import os
from pathlib import Path
from importlib import import_module

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

COLLECTION_NAME = os.getenv("POLICIES_TEXT_COLLECTION", "policies")
CHUNK_SIZE = int(os.getenv("POLICIES_TEXT_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("POLICIES_TEXT_CHUNK_OVERLAP", "150"))


def _backend_root() -> Path:
    return Path(__file__).resolve().parent


def _firebase_key_path() -> str | None:
    env_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
    if env_path:
        return str(Path(env_path).expanduser())

    # Auto-detect one Firebase Admin key file in backend/ for local dev.
    matches = sorted(_backend_root().glob("*firebase-adminsdk*.json"))
    if len(matches) == 1:
        return str(matches[0])

    return None


def _initialize_firestore_client():
    key_path = _firebase_key_path()
    if not key_path:
        raise RuntimeError(
            "Firebase service-account JSON not found. Set FIREBASE_SERVICE_ACCOUNT_PATH, "
            "or place exactly one '*firebase-adminsdk*.json' file in backend/."
        )

    key_file = Path(key_path)
    if not key_file.exists():
        raise RuntimeError(f"Firebase key file not found: {key_file}")

    try:
        firebase_admin = import_module("firebase_admin")
        firebase_credentials = import_module("firebase_admin.credentials")
        firebase_firestore = import_module("firebase_admin.firestore")
    except ImportError as import_error:
        raise RuntimeError(
            "firebase-admin is not installed. Install dependencies with `pip install -r requirements.txt`."
        ) from import_error

    if not firebase_admin._apps:
        cred = firebase_credentials.Certificate(str(key_file))
        firebase_admin.initialize_app(cred)

    print(f"Using Firebase service-account key: {key_file.name}")
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


def _safe_bool(value) -> bool:
    return bool(value)


def _safe_timestamp(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return _safe_str(value)
    return _safe_str(value)


def _base_meta(record: dict) -> dict:
    doc_id = _safe_str(record.get("document_id")) or _safe_str(record.get("id"))
    title = _safe_str(record.get("title")) or doc_id or "Untitled Source"

    return {
        "document_id": doc_id,
        "title": title,
        "category": _safe_str(record.get("category")),
        "domain": _safe_str(record.get("domain")),
        "url": _safe_str(record.get("url")),
        "effective_date": _safe_str(record.get("effective_date")),
        "has_structure": _safe_bool(record.get("has_structure", False)),
        "last_revised": _safe_str(record.get("last_revised")),
        "last_updated": _safe_timestamp(record.get("last_updated")),
        "relevant": _safe_bool(record.get("relevant", False)),
        "score": _safe_int(record.get("score", 0)),
    }


def _build_text_chunks(record: dict, base_meta: dict) -> list[Document]:
    chunks: list[Document] = []
    sections = record.get("sections") if isinstance(record.get("sections"), list) else []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for idx, section in enumerate(sections):
        if not isinstance(section, dict):
            continue

        text = _safe_str(section.get("text"))
        if not text:
            continue

        section_title = _safe_str(section.get("section_title")) or f"Section {idx + 1}"
        text_parts = splitter.split_text(text)
        for part_idx, part in enumerate(text_parts):
            if not part.strip():
                continue

            is_multi_part = len(text_parts) > 1
            source_key = section_title
            if is_multi_part:
                source_key = f"{section_title} (part {part_idx + 1}/{len(text_parts)})"

            chunks.append(
                Document(
                    page_content=part,
                    metadata={
                        **base_meta,
                        "chunk_type": "text",
                        "section_title": section_title,
                        "subsection_title": "",
                        "source_key": source_key,
                        "chunk_index": part_idx,
                        "chunk_total": len(text_parts),
                    },
                )
            )

    return chunks


def load_chunks_from_firestore() -> list[Document]:
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

    return chunks


def main() -> None:
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not set in .env")

    print(f"Loading data from Firestore collection '{COLLECTION_NAME}' ...")
    chunks = load_chunks_from_firestore()
    text_count = sum(1 for c in chunks if (c.metadata or {}).get("chunk_type") == "text")

    print(f"  {len(chunks)} total chunks ready to upsert")
    print(f"  text chunks: {text_count}")

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

    print(f"Done - {len(chunks)} text chunks indexed.")


if __name__ == "__main__":
    main()
