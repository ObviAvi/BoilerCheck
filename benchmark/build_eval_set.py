"""
Parse rag_mock_data.json and print all chunk IDs so you can build the eval set.

Chunk ID format:  {document_id}__{section_title}__{subsection_title}
(lowercased, spaces → hyphens)

Run:  python benchmark/build_eval_set.py
"""

import json
from pathlib import Path


def slugify(text: str) -> str:
    return text.lower().replace(" ", "-").replace("/", "-").replace("—", "-")


def load_chunks(data_path: str = "data/rag_mock_data.json"):
    """Return list of (chunk_id, text) tuples mirroring ingest.py logic."""
    with open(data_path) as f:
        documents = json.load(f)

    chunks = []
    for doc in documents:
        doc_id = doc["document_id"]
        for section in doc["sections"]:
            sec_slug = slugify(section["section_title"])
            if "subsections" in section:
                for sub in section["subsections"]:
                    sub_slug = slugify(sub["section_title"])
                    chunk_id = f"{doc_id}__{sec_slug}__{sub_slug}"
                    chunks.append((chunk_id, sub["text"]))
            elif "text" in section:
                chunk_id = f"{doc_id}__{sec_slug}"
                chunks.append((chunk_id, section["text"]))
    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}\n")
    for cid, text in chunks:
        preview = text[:80].replace("\n", " ")
        print(f"  {cid}")
        print(f"    → {preview}...")
        print()
