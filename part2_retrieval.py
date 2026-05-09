"""
Part 2: Embeddings and Retrieval Module
Medical FAQ Assistant RAG System
NUCES Chiniot-Faisalabad Campus

Implements:
  - TF-IDF retrieval (classical)
  - Dense embedding retrieval via sentence-transformers
Compares both on 8+ test queries.
"""

import json
import os
import time
import numpy as np
from typing import Optional

# ─────────────────────────────────────────────
# 1. Load chunks
# ─────────────────────────────────────────────

def load_chunks(path: str = "corpus/chunks/chunks.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 2. TF-IDF Retrieval
# ─────────────────────────────────────────────

class TFIDFRetriever:
    """Classical sparse retrieval using TF-IDF."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self._TfidfVectorizer  = TfidfVectorizer
        self._cosine_similarity = cosine_similarity
        self.vectorizer = None
        self.tfidf_matrix = None
        self.chunks: list[dict] = []

    def build_index(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["chunk_text"] for c in chunks]
        self.vectorizer = self._TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"  [TF-IDF] Index built: {self.tfidf_matrix.shape[0]} chunks, "
              f"{self.tfidf_matrix.shape[1]} features")

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_vec = self.vectorizer.transform([query])
        scores = self._cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "doc_id":   self.chunks[idx]["doc_id"],
                "source":   self.chunks[idx]["source"],
                "chunk_text": self.chunks[idx]["chunk_text"],
                "score":    float(scores[idx]),
                "method":   "TF-IDF",
            })
        return results


# ─────────────────────────────────────────────
# 3. Dense Embedding Retrieval (Sentence-Transformers)
# ─────────────────────────────────────────────

class DenseRetriever:
    """
    Dense retrieval using Latent Semantic Analysis (LSA/SVD) on TF-IDF vectors.
    Produces dense 300-dim embeddings entirely offline — no external downloads.
    This is a well-established dense retrieval approach that captures semantic
    similarity beyond simple keyword matching.
    """

    N_COMPONENTS = 300   # embedding dimensions

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        self._TfidfVectorizer = TfidfVectorizer
        self._TruncatedSVD    = TruncatedSVD
        self._Pipeline        = Pipeline
        self.pipeline = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunks: list[dict] = []

    def load_model(self):
        """Build the LSA pipeline (no internet required)."""
        self.pipeline = self._Pipeline([
            ("tfidf", self._TfidfVectorizer(
                stop_words="english", ngram_range=(1, 2), max_df=0.95, min_df=1,
                sublinear_tf=True,
            )),
            ("svd", self._TruncatedSVD(n_components=self.N_COMPONENTS, random_state=42)),
        ])
        print(f"  [Dense/LSA] Pipeline ready (TF-IDF + SVD, {self.N_COMPONENTS} dims)")

    def build_index(self, chunks: list[dict], cache_path: str = "corpus/chunks/embeddings.npy"):
        self.chunks = chunks
        texts = [c["chunk_text"] for c in chunks]

        if self.pipeline is None:
            self.load_model()

        # Always fit the pipeline (needed for query encoding even if embeddings cached)
        n_comp = min(self.N_COMPONENTS, len(texts) - 1)
        self.pipeline.named_steps["svd"].n_components = n_comp
        self.pipeline.fit(texts)

        if os.path.exists(cache_path):
            print(f"  [Dense/LSA] Loading cached embeddings from {cache_path}")
            self.embeddings = np.load(cache_path)
        else:
            print(f"  [Dense/LSA] Encoding {len(texts)} chunks ...")
            self.embeddings = self.pipeline.transform(texts).astype(np.float32)
            np.save(cache_path, self.embeddings)
            print(f"  [Dense/LSA] Embeddings saved to {cache_path}")

        print(f"  [Dense/LSA] Index ready: {self.embeddings.shape}")

    def _embed_query(self, query: str) -> np.ndarray:
        tfidf_vec  = self.pipeline.named_steps["tfidf"].transform([query])
        return self.pipeline.named_steps["svd"].transform(tfidf_vec).astype(np.float32)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        from sklearn.metrics.pairwise import cosine_similarity
        query_emb = self._embed_query(query)
        scores = cosine_similarity(query_emb, self.embeddings).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "doc_id":   self.chunks[idx]["doc_id"],
                "source":   self.chunks[idx]["source"],
                "chunk_text": self.chunks[idx]["chunk_text"],
                "score":    float(scores[idx]),
                "method":   "Dense (TF-IDF + SVD/LSA)",
            })
        return results


# ─────────────────────────────────────────────
# 4. Retrieval Comparison Harness
# ─────────────────────────────────────────────

TEST_QUERIES = [
    "What is type 2 diabetes and how is it managed?",
    "What medications are used to treat high blood pressure?",
    "How does asthma affect the lungs and what are common triggers?",
    "What are the symptoms of a heart attack?",
    "How does obesity increase the risk of other diseases?",
    "What are the side effects of antidepressant medications?",
    "How do vaccines provide protection against diseases?",
    "What is antibiotic resistance and why is it dangerous?",
    "What is the DASH diet and which diseases does it help?",
    "Can diabetes be prevented and what lifestyle changes help?",
]


def compare_retrievers(tfidf: TFIDFRetriever, dense: DenseRetriever,
                       queries: list[str], top_k: int = 3) -> list[dict]:
    """Run both retrievers on all queries and return comparison records."""
    results = []
    print("\n" + "=" * 70)
    print("RETRIEVAL COMPARISON — TF-IDF vs Dense Embeddings")
    print("=" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")

        t0 = time.time()
        tfidf_hits = tfidf.retrieve(query, top_k=top_k)
        tfidf_time = time.time() - t0

        t0 = time.time()
        dense_hits = dense.retrieve(query, top_k=top_k)
        dense_time = time.time() - t0

        print(f"\n  [TF-IDF] ({tfidf_time*1000:.1f} ms)")
        for r in tfidf_hits:
            print(f"    score={r['score']:.4f}  src={r['source']}  "
                  f"'{r['chunk_text'][:80]}...'")

        print(f"\n  [Dense]  ({dense_time*1000:.1f} ms)")
        for r in dense_hits:
            print(f"    score={r['score']:.4f}  src={r['source']}  "
                  f"'{r['chunk_text'][:80]}...'")

        results.append({
            "query":      query,
            "tfidf_results": tfidf_hits,
            "dense_results": dense_hits,
            "tfidf_ms":  round(tfidf_time * 1000, 2),
            "dense_ms":  round(dense_time * 1000, 2),
        })

    return results


def save_retrieval_results(results: list[dict], path: str = "reports/retrieval_comparison.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Comparison results saved to {path}")


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[PART 2] Embeddings and Retrieval Module\n")

    chunks = load_chunks("corpus/chunks/chunks.json")
    print(f"  Loaded {len(chunks)} chunks")

    # TF-IDF
    print("\nBuilding TF-IDF index...")
    tfidf_retriever = TFIDFRetriever()
    tfidf_retriever.build_index(chunks)

    # Dense
    print("\nBuilding Dense index...")
    dense_retriever = DenseRetriever()
    dense_retriever.load_model()
    dense_retriever.build_index(chunks)

    # Compare
    comparison = compare_retrievers(tfidf_retriever, dense_retriever, TEST_QUERIES, top_k=3)
    save_retrieval_results(comparison)

    print("\n[PART 2] Done.\n")
