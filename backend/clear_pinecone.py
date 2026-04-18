"""
Clear all vectors from the configured Pinecone index/namespace.

Defaults:
- Index: PINECONE_INDEX_NAME from .env
- Namespace: PINECONE_NAMESPACE from .env (empty means default namespace)

Usage:
    python clear_pinecone.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()


def main() -> None:
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    namespace = os.getenv("PINECONE_NAMESPACE", "")

    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not set in .env")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set in .env")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    print(
        f"Clearing Pinecone index '{index_name}' in namespace "
        f"'{namespace or '(default)'}' ..."
    )
    index.delete(delete_all=True, namespace=namespace)
    print("Done.")


if __name__ == "__main__":
    main()
