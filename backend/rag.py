"""
Core RAG pipeline for BoilerCheck.

Retrieves relevant Purdue policy chunks from Pinecone, reranks them with a
cross-encoder, then generates a grounded answer via the Gemini API.

Can also be run directly as a CLI:
    python rag.py "your question here" [--debug]
"""

import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

load_dotenv()

# Load once at import time so every request reuses the same in-memory models
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _doc_label(d) -> str:
    md = d.metadata or {}
    key = md.get("source_key", "")
    return key if key else d.page_content[:60] + "..."


def _print_ranking(original_docs, scores, top_n: int) -> None:
    ranked = sorted(
        zip(scores, range(len(original_docs)), original_docs),
        key=lambda x: x[0],
        reverse=True,
    )
    print("\n" + "=" * 75)
    print("RANKING: Vector Search  →  Cross-Encoder Rerank")
    print("=" * 75)
    print(f"\n{'Rerank #':<10} {'Vector #':<10} {'Score':<10} {'Status':<12} Source")
    print("-" * 75)
    for new_rank, (score, orig_idx, d) in enumerate(ranked, 1):
        kept = new_rank <= top_n
        status = "KEPT" if kept else "FILTERED"
        label = _doc_label(d)
        line = f"  {new_rank:<8} {orig_idx + 1:<10} {score:<10.4f} {status:<12} {label}"
        if not kept:
            line = f"\033[90m{line}\033[0m"
        print(line)
    print()


def query(question: str, top_k: int = 4, debug: bool = False) -> dict:
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
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not set in .env")

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=_embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        text_key="text",
    )

    candidates = vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(question)

    pairs = [[question, d.page_content] for d in candidates]
    scores = _cross_encoder.predict(pairs)

    if debug:
        _print_ranking(candidates, scores, top_n=top_k)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    docs = [d for _, d in ranked[:top_k]]

    context_parts = []
    for i, d in enumerate(docs, 1):
        md = d.metadata or {}
        url = md.get("url", "")
        section = md.get("source_key", "")
        context_parts.append(
            f"SOURCE {i}\nCITATION: [{url}#{section}]\nTEXT: {d.page_content}\n"
        )
    context = "\n".join(context_parts)

    llm = ChatGoogleGenerativeAI(
        google_api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-2.5-flash",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Purdue policy assistant. Use ONLY the sources provided. "
                "Cite using the CITATION field exactly. "
                "If there is not enough information, say you cannot confirm and ask "
                "1-2 clarifying questions. Keep your answer concise (max 4 sentences).",
            ),
            ("user", "SOURCES:\n{context}\n\nQUESTION:\n{question}"),
        ]
    )

    response = (prompt | llm).invoke({"context": context, "question": question})

    # Group retrieved chunks into document-level cards for the frontend
    docs_by_url: dict = defaultdict(lambda: {"_meta": {}, "sections": []})
    for d in docs:
        md = d.metadata or {}
        url = md.get("url", "")
        if not docs_by_url[url]["_meta"]:
            docs_by_url[url]["_meta"] = md
        section_title = (
            md.get("subsection_title")
            or md.get("section_title")
            or md.get("source_key", "")
        )
        docs_by_url[url]["sections"].append(
            {"section_title": section_title, "text": d.page_content}
        )

    documents = []
    for url, data in docs_by_url.items():
        meta = data["_meta"]
        documents.append(
            {
                "document_id": meta.get("document_id", url),
                "title": meta.get("title", ""),
                "domain": meta.get("domain", ""),
                "url": url,
                "effective_date": meta.get("effective_date", ""),
                "sections": data["sections"],
            }
        )

    return {"answer": response.content, "documents": documents}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question here" [--debug]')
        raise SystemExit(1)

    show_debug = "--debug" in sys.argv
    user_question = next(a for a in sys.argv[1:] if a != "--debug")
    result = query(user_question, debug=show_debug)

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES USED:")
    for doc in result["documents"]:
        for section in doc["sections"]:
            print(f"- {doc['url']} | {section['section_title']}")
