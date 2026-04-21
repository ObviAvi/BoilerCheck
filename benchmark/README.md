# Retrieval Benchmark

Benchmarking suite for evaluating embedding model + reranker combinations on a 20-query gold-standard eval set over Purdue housing and dining policy documents.

## How to Run

```bash
# Run all embedding x reranker combinations
python benchmark/runner.py --mode all

# Sweep embeddings only (fixed reranker)
python benchmark/runner.py --mode pass1

# Sweep rerankers only (fixed embedding)
python benchmark/runner.py --mode pass2 --embed minilm-l6-v2

# Analyze results
python benchmark/analyze_results.py --sort mrr
```

## Metrics

| Metric | What It Measures |
|---|---|
| **Hit Rate@k** | Did at least one relevant doc appear in the top-k? |
| **MRR@k** | Reciprocal rank of the first relevant doc (higher = relevant doc ranked closer to #1) |
| **NDCG@k** | Overall ranking quality — rewards having all relevant docs ranked higher |
| **Context Precision@k** | What fraction of top-k results are actually relevant? |

## Results

### Embedding Models

All models were tested with the cross-encoder reranker. Every embedding model achieved the same accuracy: **90% hit rate, 0.875 MRR@4**. The embedding model choice didn't matter for accuracy on this dataset — the main difference was latency and indexing speed.

| Embedding Model | Dim | Hit Rate@4 | MRR@4 | Avg Query Latency | Doc Embed Time |
|---|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | 0.90 | 0.875 | 231 ms | 31s |
| BAAI/bge-m3 | 1024 | 0.90 | 0.875 | 294 ms | 1447s |
| nomic-embed-text-v1 | 768 | 0.90 | 0.875 | 364 ms | 480s |
| Qwen3-Embedding-0.6B | 1024 | 0.90 | 0.875 | 545 ms | 1142s |
| gemini-embedding-001 | 768 | 0.90 | 0.875 | 782 ms | 4s |

MiniLM was the fastest for query latency and over 10x faster than most other models for document indexing (31s vs 480-1447s), while achieving identical accuracy.

### Rerankers

All rerankers were tested with the MiniLM embedding model. Results were more interesting here.

| Reranker | Hit Rate@4 | MRR@4 | Avg Query Latency |
|---|---|---|---|
| **No Reranker** | 0.90 | 0.875 | 22 ms |
| Cross-Encoder (ms-marco-MiniLM) | 0.90 | 0.875 | 231 ms |
| Cohere Rerank v3.5 | 0.95 | 0.842 | 6710 ms |
| RankLLM (Gemini 2.5 Flash) | **1.00** | **1.000** | 6570 ms |

The cross-encoder reranker and no reranker both had 90% hit rate, but they missed different queries in the eval set. The cross-encoder found "Can I have a pet in my dorm room?" (by surfacing the `__pets` chunk from rank 5+) but couldn't get "How much does the unlimited meal plan cost?" — and vice versa for no reranker. Net effect: 0% aggregate improvement.

Cohere Rerank v3.5 had a higher hit rate of 95% and found the unlimited meal plan query, but adds ~6.5s latency per query due to rate limiting on the trial API key (a paid key would eliminate this).

RankLLM using Gemini 2.5 Flash was the most accurate with 100% hit rate and perfect 1.0 MRR, but at ~6.5s per query it's too slow for a responsive user experience.

## Conclusion

**Best combo: MiniLM + No Reranker.** The cross-encoder reranker had the same accuracy as no reranker, meaning the initial vector retrieval was already good enough for this dataset. RankLLM had the best accuracy but the latency cost isn't justified. If we wanted to invest in better accuracy without the latency hit, paying for a Cohere API key (to eliminate rate limiting) would be the most practical next step.
