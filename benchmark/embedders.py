"""
Unified embedding interface for all models.
"""

import os
import time
from typing import List, Optional

import numpy as np


class Embedder:
    """Wraps different embedding backends behind a common interface."""

    def __init__(self, model_key: str, model_cfg: dict):
        self.key = model_key
        self.cfg = model_cfg
        self.model_type = model_cfg["type"]
        self.dim = model_cfg["dim"]
        self._model = None

    def _load(self):
        if self._model is not None:
            return

        if self.model_type == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            kwargs = {}
            if self.cfg.get("trust_remote_code"):
                kwargs["trust_remote_code"] = True
            self._model = SentenceTransformer(self.cfg["name"], **kwargs)

        elif self.model_type == "openai":
            import openai

            self._model = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        elif self.model_type == "google":
            from google import genai

            self._model = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        else:
            raise ValueError(f"Unknown embedding type: {self.model_type}")

    def embed_texts(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Embed a list of texts. Returns (N, dim) float32 array."""
        self._load()

        if self.model_type == "sentence-transformers":
            prefix = ""
            if is_query and self.cfg.get("query_prefix"):
                prefix = self.cfg["query_prefix"]
            elif not is_query and self.cfg.get("prompt_prefix"):
                prefix = self.cfg["prompt_prefix"]

            if prefix:
                texts = [prefix + t for t in texts]

            embeddings = self._model.encode(texts, show_progress_bar=False,
                                            normalize_embeddings=True)
            return np.array(embeddings, dtype=np.float32)

        elif self.model_type == "openai":
            # OpenAI API - batch in chunks of 2048
            all_embeddings = []
            batch_size = 2048
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = self._model.embeddings.create(
                    model=self.cfg["name"], input=batch
                )
                batch_embs = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embs)
            return np.array(all_embeddings, dtype=np.float32)

        elif self.model_type == "google":
            all_embeddings = []
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                result = self._model.models.embed_content(
                    model=self.cfg["name"],
                    contents=batch,
                )
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
            return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (dim,) array."""
        return self.embed_texts([query], is_query=True)[0]

    def embed_documents(self, docs: List[str]) -> np.ndarray:
        """Embed documents. Returns (N, dim) array."""
        return self.embed_texts(docs, is_query=False)
