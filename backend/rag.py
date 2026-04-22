"""
Core RAG pipeline for BoilerCheck.

Retrieves relevant Purdue policy chunks from Pinecone and generates a
grounded answer via the Gemini API.

No reranking: benchmarks showed MiniLM cosine search already achieves 90%
hit rate and 0.875 MRR@4. Cross-encoder reranking showed 0% improvement in
both hit rate and MRR. LLM-based reranking (RankLLM-Gemini) improved MRR to
1.0 but adds ~6.5s latency per query, not justified for this use case.
See benchmark/README.md for full analysis.

Can also be run directly as a CLI:
    python rag.py "your question here"
"""

import os
import sys
from collections import defaultdict
from typing import Iterator

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Load once at import time so every request reuses the same in-memory model
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_IMAGE_SCORE_THRESHOLD = float(os.getenv("IMAGE_SCORE_THRESHOLD", "0.35"))
_IMAGE_TOP_K = int(os.getenv("IMAGE_TOP_K", "4"))
_RETRIEVE_K = int(os.getenv("RAG_RETRIEVE_K", "8"))


# --- Cross-encoder helpers (commented out — no reranking, see benchmark/README.md) ---
# def _doc_label(d) -> str:
#     md = d.metadata or {}
#     key = md.get("source_key", "")
#     return key if key else d.page_content[:60] + "..."
#
#
# def _safe_float(value, default: float = 0.0) -> float:
#     try:
#         return float(value)
#     except (TypeError, ValueError):
#         return default
#
#
# def _normalized_score(value: float) -> float:
#     """Map cross-encoder logits to a 0-1 display score via sigmoid."""
#     x = _safe_float(value)
#     if x >= 0:
#         z = math.exp(-x)
#         return 1.0 / (1.0 + z)
#     z = math.exp(x)
#     return z / (1.0 + z)
#
#
# def _print_ranking(original_docs, scores, top_n: int) -> None:
#     ranked = sorted(
#         zip(scores, range(len(original_docs)), original_docs),
#         key=lambda x: x[0],
#         reverse=True,
#     )
#     print("\n" + "=" * 75)
#     print("RANKING: Vector Search  →  Cross-Encoder Rerank")
#     print("=" * 75)
#     print(f"\n{'Rerank #':<10} {'Vector #':<10} {'Score':<10} {'Status':<12} Source")
#     print("-" * 75)
#     for new_rank, (score, orig_idx, d) in enumerate(ranked, 1):
#         kept = new_rank <= top_n
#         status = "KEPT" if kept else "FILTERED"
#         label = _doc_label(d)
#         line = f"  {new_rank:<8} {orig_idx + 1:<10} {score:<10.4f} {status:<12} {label}"
#         if not kept:
#             line = f"\033[90m{line}\033[0m"
#         print(line)
#     print()


def _is_image_chunk(d) -> bool:
    return (d.metadata or {}).get("chunk_type") == "image"


_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Purdue policy assistant (assume all questions are related to Purdue, even if not explicitly stated). \n"
            "Use ONLY the sources provided.\n\n"
            "Structure (required):\n"
            "1) Start with **one short overview paragraph** (2–3 sentences) that directly answers the question "
            "in plain language. No bullets in this paragraph.\n"
            "2) Then a blank line, then a GitHub-flavored Markdown **bullet list**: each supporting point on its "
            "own line starting with `- `. One clear fact per bullet.\n\n"
            "Citations: Each block in SOURCES is labeled `SOURCE n`. At the **end** of the overview paragraph "
            "and at the **end** of each bullet, add plain square-bracket source markers only—e.g. `[1]`, `[2]`—"
            "matching the SOURCE number you relied on. Reuse the same number when the same SOURCE applies. "
            "Do **not** use markdown links, URLs, or footnotes; only these numeric markers.\n\n"
            "Never write `[CITATION]` or other placeholders.\n\n"
            "If there is not enough information, say you cannot confirm and ask 1–2 clarifying questions.\n\n"
            "Stay concise (overview + typically 3–6 bullets).",
        ),
        ("user", "SOURCES:\n{context}\n\nQUESTION:\n{question}"),
    ]
)


def retrieve(question: str, top_k: int = 4) -> tuple[str, list]:
    """
    Pinecone vector retrieval; build LLM context string and document cards.

    Uses cosine similarity scores directly from Pinecone (no reranking).

    Returns:
        (context, documents) in the same shape as the /ask JSON payload.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not set in .env")

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=_embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        text_key="text",
    )

    results = vectorstore.similarity_search_with_relevance_scores(question, k=_RETRIEVE_K)

    if not results:
        return "", []

    top_text = []
    passing_images = []
    for d, score in results:
        if _is_image_chunk(d):
            if score >= _IMAGE_SCORE_THRESHOLD:
                passing_images.append((score, d))
            continue

        if len(top_text) < top_k:
            top_text.append((score, d))

    if not top_text and results:
        top_text = [(score, d) for d, score in results[:top_k]]

    selected_images = passing_images[:_IMAGE_TOP_K]
    selected = top_text + selected_images

    context_parts = []
    for i, (score, d) in enumerate(selected, 1):
        md = d.metadata or {}
        url = md.get("url", "") or md.get("image_source_url", "")
        section = md.get("source_key", "")
        source_type = md.get("chunk_type", "text")
        context_parts.append(
            f"SOURCE {i}\n"
            f"TYPE: {source_type}\n"
            f"SCORE: {score:.4f}\n"
            f"CITATION: [{url}#{section}]\n"
            f"TEXT: {d.page_content}\n"
        )
    context = "\n".join(context_parts)

    docs_by_id: dict = defaultdict(lambda: {"_meta": {}, "sections": [], "images": []})
    for score, d in selected:
        md = d.metadata or {}
        doc_key = md.get("document_id") or md.get("url") or md.get("image_source_url") or "unknown"
        if not docs_by_id[doc_key]["_meta"]:
            docs_by_id[doc_key]["_meta"] = md

        if _is_image_chunk(d):
            image_url = md.get("image_source_url", "")
            image_entry = {
                "description": md.get("image_description", d.page_content),
                "filename": md.get("image_filename", ""),
                "format": md.get("image_format", ""),
                "image_type": md.get("image_type", ""),
                "md5": md.get("image_md5", ""),
                "source_url": image_url,
                "public_url": md.get("image_public_url", ""),
                "width": int(md.get("image_width", 0) or 0),
                "height": int(md.get("image_height", 0) or 0),
                "score": round(score, 4),
            }

            dedupe_key = (
                image_entry["source_url"],
                image_entry["md5"],
                image_entry["filename"],
            )
            existing = {
                (img.get("source_url", ""), img.get("md5", ""), img.get("filename", ""))
                for img in docs_by_id[doc_key]["images"]
            }
            if dedupe_key not in existing:
                docs_by_id[doc_key]["images"].append(image_entry)
            continue

        section_title = (
            md.get("subsection_title")
            or md.get("section_title")
            or md.get("source_key", "")
        )
        docs_by_id[doc_key]["sections"].append(
            {
                "section_title": section_title,
                "text": d.page_content,
                "score": round(score, 4),
            }
        )

    documents = []
    for _, data in docs_by_id.items():
        meta = data["_meta"]
        images = sorted(data["images"], key=lambda img: img.get("score", 0), reverse=True)
        doc_url = meta.get("url", "")
        if not doc_url and images:
            doc_url = images[0].get("source_url", "")
        documents.append(
            {
                "document_id": meta.get("document_id", ""),
                "title": meta.get("title", "") or meta.get("document_id", "Untitled Source"),
                "domain": meta.get("domain", ""),
                "url": doc_url,
                "effective_date": meta.get("effective_date", ""),
                "sections": data["sections"],
                "has_structure": bool(meta.get("has_structure", False)),
                "images": images,
            }
        )

    return context, documents


def query(question: str, top_k: int = 4) -> dict:
    """
    Run the full RAG pipeline and return a structured response.

    Returns:
        {
            "answer": str,
            "documents": [
                {
                    "document_id": str,
                    "title": str,
                    "domain": str,
                    "url": str,
                    "effective_date": str,
                    "sections": [{"section_title": str, "text": str}, ...]
                },
                ...
            ]
        }
    """
    context, documents = retrieve(question, top_k=top_k)

    llm = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash",
        temperature=0,
    )

    response = (_RAG_PROMPT | llm).invoke({"context": context, "question": question})

    return {"answer": response.content, "documents": documents}


def stream_rag_events(question: str, top_k: int = 4) -> Iterator[dict]:
    """
    Yields events for SSE: documents first, then token chunks, then done.

    Event shapes:
        {"type": "documents", "documents": [...]}
        {"type": "token", "text": str}
        {"type": "done"}
    """
    context, documents = retrieve(question, top_k=top_k)
    yield {"type": "documents", "documents": documents}

    llm = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash-lite",
        temperature=0,
        streaming=True,
    )

    for chunk in (_RAG_PROMPT | llm).stream({"context": context, "question": question}):
        raw = chunk.content
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, list):
            text = "".join(
                p if isinstance(p, str) else (p.get("text", "") if isinstance(p, dict) else "")
                for p in raw
            )
        else:
            text = str(raw) if raw else ""
        if text:
            yield {"type": "token", "text": text}

    yield {"type": "done"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question here"')
        raise SystemExit(1)

    user_question = sys.argv[1]
    result = query(user_question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES USED:")
    for doc in result["documents"]:
        for section in doc["sections"]:
            print(f"- {doc['url']} | {section['section_title']}")
