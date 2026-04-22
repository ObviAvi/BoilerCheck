"""
Main benchmark orchestrator.

Usage:
    python benchmark/runner.py --pass 1                    # Sweep all embeddings (fixed reranker)
    python benchmark/runner.py --pass 2 --embedding nomic-v1  # Sweep all rerankers (fixed embedding)
    python benchmark/runner.py --embedding minilm-l6-v2 --reranker cross-encoder-minilm  # Single combo
    python benchmark/runner.py --all                       # Run all embedding × reranker combos
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# .env lives in the BoilerCheck repo root (parents[1]).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, os.path.dirname(__file__))

from build_eval_set import load_chunks
from config import EMBEDDING_MODELS, RERANKER_MODELS, RETRIEVE_K, RERANK_TOP_K
from embedders import Embedder
from metrics import compute_all_metrics
from rerankers import Reranker

RESULTS_DIR = Path(__file__).parent / "results"
EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set():
    with open(EVAL_SET_PATH) as f:
        return json.load(f)


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS flat L2 index from document embeddings."""
    import faiss

    dim = embeddings.shape[1]
    # Inner product — embeddings are L2-normalized so this is cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_from_faiss(index, query_embedding: np.ndarray, k: int):
    """Retrieve top-k document indices from FAISS index."""
    query_embedding = query_embedding.reshape(1, -1)
    scores, indices = index.search(query_embedding, k)
    return indices[0].tolist(), scores[0].tolist()


def run_single_combo(
    embedding_key: str,
    reranker_key: str,
    chunks: list,
    eval_set: list,
    verbose: bool = True,
) -> dict:
    """Run benchmark for one embedding + reranker combination."""
    emb_cfg = EMBEDDING_MODELS[embedding_key]
    rer_cfg = RERANKER_MODELS[reranker_key]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Embedding: {embedding_key} ({emb_cfg['name']})")
        print(f"  Reranker:  {reranker_key} ({rer_cfg['name']})")
        print(f"{'='*70}")

    chunk_ids = [c[0] for c in chunks]
    chunk_texts = [c[1] for c in chunks]

    embedder = Embedder(embedding_key, emb_cfg)

    if verbose:
        print("  Embedding documents...", end=" ", flush=True)
    t0 = time.time()
    doc_embeddings = embedder.embed_documents(chunk_texts)
    embed_doc_time = time.time() - t0
    if verbose:
        print(f"done ({embed_doc_time:.1f}s)")

    if verbose:
        print("  Building FAISS index...", end=" ", flush=True)
    index = build_faiss_index(doc_embeddings)
    if verbose:
        print("done")

    reranker = Reranker(reranker_key, rer_cfg)

    query_results = []
    total_latency_ms = 0

    for qi, q in enumerate(eval_set):
        query = q["query"]
        relevant_ids = set(q["relevant_chunk_ids"])

        t0 = time.time()
        q_emb = embedder.embed_query(query)
        embed_query_time = time.time() - t0

        t0 = time.time()
        indices, scores = retrieve_from_faiss(index, q_emb, RETRIEVE_K)
        retrieve_time = time.time() - t0

        retrieved_ids = [chunk_ids[i] for i in indices if i < len(chunk_ids)]
        retrieved_texts = [chunk_texts[i] for i in indices if i < len(chunk_ids)]

        t0 = time.time()
        reranked = reranker.rerank(query, retrieved_texts, retrieved_ids, RERANK_TOP_K)
        rerank_time = time.time() - t0

        final_ids = [did for did, _ in reranked]
        latency_ms = (embed_query_time + retrieve_time + rerank_time) * 1000
        total_latency_ms += latency_ms

        metrics = compute_all_metrics(final_ids, relevant_ids, RERANK_TOP_K)
        metrics["latency_ms"] = round(latency_ms, 2)
        metrics["query"] = query
        metrics["retrieved_ids"] = final_ids

        query_results.append(metrics)

        if verbose:
            hit = "HIT" if metrics["hit_rate@k"] > 0 else "MISS"
            print(f"  [{qi+1:2d}/{len(eval_set)}] {hit}  MRR={metrics['mrr@k']:.2f}  "
                  f"P@k={metrics['context_precision@k']:.2f}  {latency_ms:6.1f}ms  {query[:50]}")

    n = len(query_results)
    agg = {
        "hit_rate@k": sum(r["hit_rate@k"] for r in query_results) / n,
        "mrr@k": sum(r["mrr@k"] for r in query_results) / n,
        "ndcg@k": sum(r["ndcg@k"] for r in query_results) / n,
        "context_precision@k": sum(r["context_precision@k"] for r in query_results) / n,
        "avg_latency_ms": round(total_latency_ms / n, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "embed_doc_time_s": round(embed_doc_time, 2),
    }

    result = {
        "embedding": embedding_key,
        "embedding_model": emb_cfg["name"],
        "reranker": reranker_key,
        "reranker_model": rer_cfg["name"],
        "settings": {"retrieve_k": RETRIEVE_K, "rerank_top_k": RERANK_TOP_K},
        "aggregate_metrics": agg,
        "per_query": query_results,
        "timestamp": datetime.now().isoformat(),
    }

    if verbose:
        print(f"\n  --- Aggregate ---")
        print(f"  Hit Rate@{RERANK_TOP_K}:        {agg['hit_rate@k']:.3f}")
        print(f"  MRR@{RERANK_TOP_K}:             {agg['mrr@k']:.3f}")
        print(f"  NDCG@{RERANK_TOP_K}:            {agg['ndcg@k']:.3f}")
        print(f"  Context Prec@{RERANK_TOP_K}:    {agg['context_precision@k']:.3f}")
        print(f"  Avg Latency:          {agg['avg_latency_ms']:.1f} ms")
        print(f"  Doc Embedding Time:   {agg['embed_doc_time_s']:.1f} s")

    return result


def save_result(result: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    fname = f"{result['embedding']}__{result['reranker']}.json"
    path = RESULTS_DIR / fname
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="BoilerCheck RAG Benchmark")
    parser.add_argument("--pass", dest="bench_pass", type=int, choices=[1, 2],
                        help="Pass 1: sweep embeddings. Pass 2: sweep rerankers.")
    parser.add_argument("--embedding", type=str, help="Specific embedding model key")
    parser.add_argument("--reranker", type=str, help="Specific reranker model key")
    parser.add_argument("--all", action="store_true", help="Run all combos")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    chunks = load_chunks()
    eval_set = load_eval_set()
    verbose = not args.quiet

    print(f"Loaded {len(chunks)} chunks, {len(eval_set)} eval queries")
    print(f"Retrieve top-{RETRIEVE_K}, rerank to top-{RERANK_TOP_K}")

    combos = []

    if args.bench_pass == 1:
        reranker = "cross-encoder-minilm"
        for emb_key in EMBEDDING_MODELS:
            combos.append((emb_key, reranker))

    elif args.bench_pass == 2:
        if not args.embedding:
            print("ERROR: --pass 2 requires --embedding <key>")
            sys.exit(1)
        for rer_key in RERANKER_MODELS:
            combos.append((args.embedding, rer_key))

    elif args.embedding and args.reranker:
        combos.append((args.embedding, args.reranker))

    elif args.all:
        for emb_key in EMBEDDING_MODELS:
            for rer_key in RERANKER_MODELS:
                combos.append((emb_key, rer_key))
    else:
        combos.append(("minilm-l6-v2", "cross-encoder-minilm"))

    print(f"\nRunning {len(combos)} combination(s)...\n")

    for emb_key, rer_key in combos:
        try:
            result = run_single_combo(emb_key, rer_key, chunks, eval_set, verbose)
            save_result(result)
        except Exception as e:
            print(f"\n  ERROR with {emb_key} + {rer_key}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
