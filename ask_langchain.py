import os
import sys
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder

load_dotenv()

def doc_label(d):
    """Short label for a document: source_key or first 60 chars of text."""
    md = d.metadata or {}
    key = md.get("source_key", "")
    return key if key else d.page_content[:60] + "..."


def print_ranking_comparison(original_docs, scores, top_n):
    """Show all candidates with rerank scores, marking which were kept/filtered."""
    ranked = sorted(
        zip(scores, range(len(original_docs)), original_docs),
        key=lambda x: x[0], reverse=True,
    )

    print("\n" + "=" * 75)
    print("RANKING COMPARISON: Vector Search vs. Cross-Encoder Rerank")
    print("=" * 75)
    print(f"\n{'Rerank #':<10} {'Vector #':<10} {'Score':<10} {'Status':<12} Source")
    print("-" * 75)
    for new_rank, (score, orig_idx, d) in enumerate(ranked, 1):
        old_rank = orig_idx + 1
        kept = new_rank <= top_n
        status = "KEPT" if kept else "FILTERED"
        label = doc_label(d)
        line = f"  {new_rank:<8} {old_rank:<10} {score:<10.4f} {status:<12} {label}"
        if not kept:
            line = f"\033[90m{line}\033[0m"  # dim filtered rows
        print(line)

    print()


def main(question: str, debug=False):
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Pinecone
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME in .env")

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        text_key="text",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # Get original vector search results
    original_docs = retriever.invoke(question)

    # Rerank with cross-encoder for better relevance
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[question, d.page_content] for d in original_docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, original_docs), key=lambda x: x[0], reverse=True)
    docs = [d for _, d in ranked[:4]]

    if debug:
        print_ranking_comparison(original_docs, scores, top_n=4)

    # Format context with URL citations
    context_parts = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        section = md.get("source_key", "")
        url = md.get("url", "")
        citation = f"[{url}#{section}]"
        context_parts.append(
            f"SOURCE {i}\nCITATION: {citation}\nTEXT: {d.page_content}\n"
        )
    context = "\n".join(context_parts)

    llm = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free",
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Purdue policy assistant. Use ONLY the sources provided. "
                "Cite using the CITATION field exactly. If not enough info, say you cannot confirm and ask 1-2 clarifying questions. "
                "Keep it short (max 4 sentences).",
            ),
            ("user", "SOURCES:\n{context}\n\nQUESTION:\n{question}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    print("\nANSWER:")
    print(response.content)

    print("\nSOURCES USED:\n")
    seen = set()
    for d in docs:  
        md = d.metadata or {}
        url = md.get("url", "")
        section_title = md.get("section_title", "")
        subsection_title = md.get("subsection_title", "")
        section_path = section_title if not subsection_title else f"{section_title} > {subsection_title}"
        key = (url, section_path)
        if url and key not in seen:
            print(f"- {url} | {section_path}")
            seen.add(key)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/ask_langchain.py "your question here" [--debug]')
        raise SystemExit(1)

    show_debug = "--debug" in sys.argv
    query = [a for a in sys.argv[1:] if a != "--debug"][0]
    main(query, debug=show_debug)