"""
Unified reranker interface for all models.
"""

import os
import time
from typing import List, Tuple


class Reranker:
    """Wraps different reranking backends behind a common interface."""

    def __init__(self, model_key: str, model_cfg: dict):
        self.key = model_key
        self.cfg = model_cfg
        self.model_type = model_cfg["type"]
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        if self.model_type == "none":
            pass

        elif self.model_type == "cross-encoder":
            from sentence_transformers import CrossEncoder

            kwargs = {}
            if self.cfg.get("trust_remote_code"):
                kwargs["trust_remote_code"] = True
            self._model = CrossEncoder(self.cfg["name"], **kwargs)

        elif self.model_type == "qwen-reranker":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg["name"], trust_remote_code=True, padding_side="left"
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.cfg["name"], trust_remote_code=True, torch_dtype=torch.float32,
                num_labels=1,
            )
            self._model.config.pad_token_id = self._tokenizer.pad_token_id
            self._model.eval()

        elif self.model_type == "cohere":
            import cohere

            self._model = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

        elif self.model_type == "rankllm":
            from google import genai

            self._model = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        else:
            raise ValueError(f"Unknown reranker type: {self.model_type}")

    def rerank(
        self, query: str, doc_texts: List[str], doc_ids: List[str], top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Rerank documents and return top_k as [(doc_id, score), ...] sorted by
        relevance descending.
        """
        self._load()

        if self.model_type == "none":
            # No reranking — return in original retrieval order with dummy scores
            return [(did, 1.0 - i * 0.01) for i, did in enumerate(doc_ids[:top_k])]

        elif self.model_type == "cross-encoder":
            pairs = [[query, text] for text in doc_texts]
            scores = self._model.predict(pairs).tolist()
            ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

        elif self.model_type == "qwen-reranker":
            return self._qwen_rerank(query, doc_texts, doc_ids, top_k)

        elif self.model_type == "cohere":
            # Rate limit: trial key allows 10 calls/min
            if hasattr(self, "_last_cohere_call"):
                elapsed = time.time() - self._last_cohere_call
                if elapsed < 7:  # ~8.5 calls/min to stay safe
                    time.sleep(7 - elapsed)
            self._last_cohere_call = time.time()

            response = self._model.rerank(
                model=self.cfg["name"],
                query=query,
                documents=doc_texts,
                top_n=top_k,
            )
            results = []
            for r in response.results:
                idx = r.index
                results.append((doc_ids[idx], r.relevance_score))
            return results

        elif self.model_type == "rankllm":
            # LLM-based listwise reranking using Gemini
            return self._rankllm_rerank(query, doc_texts, doc_ids, top_k)

    def _qwen_rerank(
        self, query: str, doc_texts: List[str], doc_ids: List[str], top_k: int
    ) -> List[Tuple[str, float]]:
        """Qwen3-Reranker using transformers directly with instruction format."""
        import torch

        task_instruction = "Given a web search query, retrieve relevant passages that answer the query"
        formatted_query = f"Instruct: {task_instruction}\nQuery: {query}"

        pairs = [[formatted_query, text] for text in doc_texts]
        scores = []

        with torch.no_grad():
            for pair in pairs:
                inputs = self._tokenizer(
                    pair, padding=True, truncation=True, max_length=512,
                    return_tensors="pt"
                )
                output = self._model(**inputs)
                score = output.logits.squeeze().item()
                scores.append(score)

        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _rankllm_rerank(
        self, query: str, doc_texts: List[str], doc_ids: List[str], top_k: int
    ) -> List[Tuple[str, float]]:
        """Use Gemini as a listwise reranker."""
        doc_list = "\n".join(
            f"[{i+1}] {text[:500]}" for i, text in enumerate(doc_texts)
        )
        prompt = (
            f"Given the query: \"{query}\"\n\n"
            f"Rank the following documents from most to least relevant. "
            f"Return ONLY a comma-separated list of document numbers in order of "
            f"relevance (e.g., \"3,1,5,2,4\"). No explanation.\n\n"
            f"Documents:\n{doc_list}"
        )

        response = self._model.models.generate_content(
            model=self.cfg["name"], contents=prompt
        )
        text = response.text.strip()

        # Parse ranking
        try:
            ranking = [int(x.strip()) - 1 for x in text.split(",")]
        except (ValueError, IndexError):
            # Fallback: return original order
            ranking = list(range(len(doc_ids)))

        results = []
        for rank, idx in enumerate(ranking[:top_k]):
            if 0 <= idx < len(doc_ids):
                score = 1.0 - rank * 0.1  # synthetic descending score
                results.append((doc_ids[idx], score))

        # Fill if parsing gave fewer than top_k
        seen = {r[0] for r in results}
        for did in doc_ids:
            if len(results) >= top_k:
                break
            if did not in seen:
                results.append((did, 0.0))
                seen.add(did)

        return results[:top_k]
