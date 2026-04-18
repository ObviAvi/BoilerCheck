"""
Ingest local mock RAG data into Pinecone.

Reads ../data/rag_mock_data.json, creates one vector entry per text section
(and optional image descriptions), embeds with all-MiniLM-L6-v2, and upserts
with deterministic vector IDs so re-runs do not create duplicates.

Usage:
    python ingest_mock_data.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mock_data_path() -> Path:
    return _project_root() / "data" / "rag_mock_data.json"


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

    return {
        "document_id": doc_id,
        "title": title,
        "domain": _safe_str(record.get("domain")),
        "url": _safe_str(record.get("url")),
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

        parent_title = _safe_str(section.get("section_title")) or f"Section {idx + 1}"

        # Format 1: flat section with direct `text`.
        direct_text = _safe_str(section.get("text"))
        if direct_text:
            chunks.append(
                Document(
                    page_content=direct_text,
                    metadata={
                        **base_meta,
                        "chunk_type": "text",
                        "section_title": parent_title,
                        "subsection_title": "",
                        "source_key": parent_title,
                    },
                )
            )

        # Format 2: nested `subsections` list.
        subsections = (
            section.get("subsections") if isinstance(section.get("subsections"), list) else []
        )
        for sub_idx, sub in enumerate(subsections):
            if not isinstance(sub, dict):
                continue

            sub_text = _safe_str(sub.get("text"))
            if not sub_text:
                continue

            subsection_title = _safe_str(sub.get("section_title")) or f"Subsection {sub_idx + 1}"
            source_key = f"{parent_title} / {subsection_title}"
            chunks.append(
                Document(
                    page_content=sub_text,
                    metadata={
                        **base_meta,
                        "chunk_type": "text",
                        "section_title": parent_title,
                        "subsection_title": subsection_title,
                        "source_key": source_key,
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


def _vector_id(doc: Document) -> str:
    meta = doc.metadata or {}
    key = "|".join(
        [
            _safe_str(meta.get("document_id")),
            _safe_str(meta.get("chunk_type")),
            _safe_str(meta.get("source_key")),
            doc.page_content,
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _load_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Mock data file not found: {path}")

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise RuntimeError("Mock data must be a JSON array of records.")

    chunks: list[Document] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        base_meta = _base_meta(record)
        chunks.extend(_build_text_chunks(record, base_meta))
        chunks.extend(_build_image_chunks(record, base_meta))

    return chunks


def _upsert_documents(index_name: str, namespace: str, docs: list[Document]) -> None:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = [d.page_content for d in docs]
    vectors = embeddings.embed_documents(texts)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]

        payload = []
        for doc, vector in zip(batch_docs, batch_vectors):
            metadata = dict(doc.metadata or {})
            metadata["text"] = doc.page_content
            payload.append(
                {
                    "id": _vector_id(doc),
                    "values": vector,
                    "metadata": metadata,
                }
            )

        index.upsert(vectors=payload, namespace=namespace)


def main() -> None:
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    namespace = os.getenv("PINECONE_NAMESPACE", "")

    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not set in .env")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in .env")

    data_path = _mock_data_path()
    print(f"Loading mock data from: {data_path}")
    docs = _load_documents(data_path)

    if not docs:
        raise RuntimeError("No chunks were produced from mock data.")

    text_count = sum(1 for d in docs if (d.metadata or {}).get("chunk_type") == "text")
    image_count = sum(1 for d in docs if (d.metadata or {}).get("chunk_type") == "image")

    print(f"Prepared {len(docs)} chunks ({text_count} text, {image_count} image).")

    print(f"Upserting into Pinecone index '{index_name}' ...")
    _upsert_documents(index_name, namespace, docs)
    print("Done.")


if __name__ == "__main__":
    main()
