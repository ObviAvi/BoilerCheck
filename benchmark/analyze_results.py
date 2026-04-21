"""
Load benchmark results and print comparison tables.

Usage:
    python benchmark/analyze_results.py                # Show all results
    python benchmark/analyze_results.py --pass 1       # Only embedding sweep results
    python benchmark/analyze_results.py --pass 2       # Only reranker sweep results
    python benchmark/analyze_results.py --sort mrr     # Sort by MRR (default: hit_rate)
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

SORT_KEYS = {
    "hit_rate": "hit_rate@k",
    "mrr": "mrr@k",
    "ndcg": "ndcg@k",
    "precision": "context_precision@k",
    "latency": "avg_latency_ms",
}


def load_all_results():
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def filter_results(results, bench_pass=None):
    if bench_pass == 1:
        return [r for r in results if r["reranker"] == "cross-encoder-minilm"]
    elif bench_pass == 2:
        # Find the most common embedding (the fixed one in pass 2)
        embeddings = [r["embedding"] for r in results if r["reranker"] != "cross-encoder-minilm"]
        if not embeddings:
            return results
        from collections import Counter
        fixed_emb = Counter(embeddings).most_common(1)[0][0]
        return [r for r in results if r["embedding"] == fixed_emb]
    return results


def print_table(results, sort_key="hit_rate@k"):
    if not results:
        print("No results found.")
        return

    # Sort (latency ascending, everything else descending)
    reverse = sort_key != "avg_latency_ms"
    results.sort(key=lambda r: r["aggregate_metrics"].get(sort_key, 0), reverse=reverse)

    # Header
    print(f"\n{'Rank':<5} {'Embedding':<22} {'Reranker':<24} "
          f"{'Hit@k':>6} {'MRR@k':>6} {'NDCG@k':>7} {'Prec@k':>7} "
          f"{'Latency':>9} {'Embed(s)':>9}")
    print("-" * 105)

    for i, r in enumerate(results, 1):
        m = r["aggregate_metrics"]
        print(f"{i:<5} {r['embedding']:<22} {r['reranker']:<24} "
              f"{m['hit_rate@k']:>6.3f} {m['mrr@k']:>6.3f} {m['ndcg@k']:>7.3f} "
              f"{m['context_precision@k']:>7.3f} {m['avg_latency_ms']:>7.1f}ms "
              f"{m['embed_doc_time_s']:>7.1f}s")

    print()

    # Best combo
    best = results[0]
    bm = best["aggregate_metrics"]
    print(f"Best: {best['embedding']} + {best['reranker']}")
    print(f"  Hit Rate: {bm['hit_rate@k']:.3f} | MRR: {bm['mrr@k']:.3f} | "
          f"NDCG: {bm['ndcg@k']:.3f} | Latency: {bm['avg_latency_ms']:.1f}ms")


def print_per_query_breakdown(results):
    """Show which queries each model got right/wrong."""
    if not results:
        return

    print(f"\n{'='*80}")
    print("PER-QUERY BREAKDOWN (queries missed by at least one model)")
    print(f"{'='*80}\n")

    # Collect all queries
    queries = [q["query"] for q in results[0]["per_query"]]

    for qi, query in enumerate(queries):
        hits = []
        misses = []
        for r in results:
            label = f"{r['embedding']}+{r['reranker']}"
            pq = r["per_query"][qi]
            if pq["hit_rate@k"] > 0:
                hits.append(label)
            else:
                misses.append(label)

        if misses:  # only show queries with at least one miss
            print(f"  Q: {query}")
            print(f"     HIT:  {', '.join(hits) if hits else '(none)'}")
            print(f"     MISS: {', '.join(misses)}")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="bench_pass", type=int, choices=[1, 2])
    parser.add_argument("--sort", default="hit_rate", choices=list(SORT_KEYS.keys()))
    parser.add_argument("--breakdown", action="store_true", help="Show per-query breakdown")
    args = parser.parse_args()

    results = load_all_results()
    if not results:
        print("No results found in benchmark/results/. Run runner.py first.")
        return

    results = filter_results(results, args.bench_pass)
    sort_key = SORT_KEYS[args.sort]

    title = "All Results"
    if args.bench_pass == 1:
        title = "Pass 1: Embedding Sweep (fixed reranker: cross-encoder-minilm)"
    elif args.bench_pass == 2:
        title = "Pass 2: Reranker Sweep"

    print(f"\n{'='*105}")
    print(f"  {title}")
    print(f"  Sorted by: {args.sort} | {len(results)} result(s)")
    print(f"{'='*105}")

    print_table(results, sort_key)

    if args.breakdown:
        print_per_query_breakdown(results)


if __name__ == "__main__":
    main()
