import json
import numpy as np
import faiss
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    """
    Hybrid retriever with:
    - FAISS semantic search
    - BM25 lexical search
    - Reciprocal Rank Fusion (RRF)
    - Section-aware reranking
    """

    def __init__(
        self,
        faiss_index_path: str,
        metadata_path: str,
        embedding_model: str = "all-mpnet-base-v2",
        rrf_k: int = 60,
    ):
        self.rrf_k = rrf_k

        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Prepare corpus for BM25
        self.documents = [item["text"] for item in self.metadata]
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Embedding model
        self.model = SentenceTransformer(embedding_model)

        # Section bias configuration (the "value system")
        self.section_bias = {
            "External links": 0.2,
            "References": 0.3,
            "See also": 0.4,
            # mildly preferred
            "Reception": 1.2,
            "Development": 1.1,
            "Gameplay": 1.1,
            "Controversy": 1.3,
        }

    # ---------- Core retrieval steps ----------

    def semantic_search(self, query: str, top_k: int):
        query_embedding = self.model.encode(query)
        query_vector = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query_vector, top_k)

        return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]

    def lexical_search(self, query: str, top_k: int):
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def reciprocal_rank_fusion(self, rankings):
        rrf_scores = defaultdict(float)

        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)

        return rrf_scores

    # ---------- Section-aware rerank ----------

    def apply_section_bias(self, scores: dict):
        adjusted = {}

        for doc_id, score in scores.items():
            section = self.metadata[doc_id].get("section")
            bias = self.section_bias.get(section, 1.0)
            adjusted[doc_id] = score * bias

        return adjusted

    # ---------- Public API ----------

    def search(
        self,
        query: str,
        faiss_top_k: int = 50,
        bm25_top_k: int = 50,
        final_top_k: int = 5,
    ):
        # Step 1: retrieve
        semantic = self.semantic_search(query, faiss_top_k)
        lexical = self.lexical_search(query, bm25_top_k)

        # Step 2: RRF
        rrf_scores = self.reciprocal_rank_fusion([semantic, lexical])

        # Step 3: section-aware rerank
        reranked_scores = self.apply_section_bias(rrf_scores)

        # Step 4: sort & format
        top_docs = sorted(
            reranked_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:final_top_k]

        results = []
        for rank, (doc_id, score) in enumerate(top_docs, start=1):
            meta = self.metadata[doc_id]
            results.append({
                "rank": rank,
                "score": score,
                "doc_id": doc_id,
                "page_title": meta.get("page_title"),
                "section": meta.get("section"),
                "text": meta.get("text"),
                "url": meta.get("url"),
            })

        return results
