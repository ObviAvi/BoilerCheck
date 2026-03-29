"""
Ingestion script for BoilerCheck.

Reads data/rag_mock_data.json, splits it into subsection-level chunks,
embeds them with all-MiniLM-L6-v2, and upserts into Pinecone.

Run once (or whenever the source data changes):
    python ingest.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

DATA_FILE = Path(__file__).parent.parent / "data" / "rag_mock_data.json"


def load_chunks(path: Path) -> list[Document]:
    """
    Flatten the nested JSON into one Document per subsection (or section
    when there are no subsections). Each Document carries the full document
    metadata so the frontend can render source cards.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    chunks: list[Document] = []

    for doc in data:
        base_meta = {
            "document_id": doc["document_id"],
            "title":        doc["title"],
            "domain":       doc["domain"],
            "url":          doc["url"],
            "effective_date": doc["effective_date"],
        }

        for section in doc["sections"]:
            section_title = section["section_title"]

            if "subsections" in section:
                for sub in section["subsections"]:
                    sub_title = sub["section_title"]
                    chunks.append(
                        Document(
                            page_content=sub["text"],
                            metadata={
                                **base_meta,
                                "section_title":    section_title,
                                "subsection_title": sub_title,
                                "source_key":       f"{section_title}/{sub_title}",
                            },
                        )
                    )
            elif "text" in section:
                chunks.append(
                    Document(
                        page_content=section["text"],
                        metadata={
                            **base_meta,
                            "section_title":    section_title,
                            "subsection_title": "",
                            "source_key":       section_title,
                        },
                    )
                )

    return chunks


def main():
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not set in .env")

    print(f"Loading data from {DATA_FILE} ...")
    chunks = load_chunks(DATA_FILE)
    print(f"  {len(chunks)} chunks ready to upsert")

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

    print(f"Done — {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    main()
