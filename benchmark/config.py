"""
Model registry and benchmark settings.
"""

# ---------------------------------------------------------------------------
# Retrieval settings (match production pipeline)
# ---------------------------------------------------------------------------
RETRIEVE_K = 8        # candidates from vector search
RERANK_TOP_K = 4      # kept after reranking

# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------
EMBEDDING_MODELS = {
    "minilm-l6-v2": {
        "name": "all-MiniLM-L6-v2",
        "type": "sentence-transformers",
        "dim": 384,
    },
    "openai-large": {
        "name": "text-embedding-3-large",
        "type": "openai",
        "dim": 3072,
    },
    "gemini-embedding": {
        "name": "models/gemini-embedding-001",
        "type": "google",
        "dim": 768,
    },
    "nomic-v1": {
        "name": "nomic-ai/nomic-embed-text-v1",
        "type": "sentence-transformers",
        "dim": 768,
        "trust_remote_code": True,
        "prompt_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "type": "sentence-transformers",
        "dim": 1024,
    },
    "qwen3-embedding": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "type": "sentence-transformers",
        "dim": 1024,
        "trust_remote_code": True,
    },
}

# ---------------------------------------------------------------------------
# Reranker model registry
# ---------------------------------------------------------------------------
RERANKER_MODELS = {
    "none": {
        "name": "No Reranker",
        "type": "none",
    },
    "cross-encoder-minilm": {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "type": "cross-encoder",
    },
    "jina-reranker-v2": {
        "name": "jinaai/jina-reranker-v2-base-multilingual",
        "type": "cross-encoder",
        "trust_remote_code": True,
    },
    "qwen3-reranker": {
        "name": "Qwen/Qwen3-Reranker-0.6B",
        "type": "qwen-reranker",
        "trust_remote_code": True,
    },
    "cohere-rerank": {
        "name": "rerank-v3.5",
        "type": "cohere",
    },
    "rankllm-gemini": {
        "name": "gemini-2.5-flash",
        "type": "rankllm",
    },
}
