from hybrid_retriever import HybridRetriever

retriever = HybridRetriever(
    faiss_index_path="../data/processed/hitman_faiss.index",
    metadata_path="../data/processed/hitman_index_mapping.json",
)

query = "Why is Hitman: Blood Money popular"

results = retriever.search(query)

print("\nTop results after Hybrid + Section-aware retrieval:")
print("=" * 80)

for r in results:
    print(f"Rank {r['rank']}, score: {r['score']:.4f}")
    print(f"Title: {r['page_title']}, Section: {r['section']}")
    print(f"Text: {r['text'][:300]}...")
    print(f"URL: {r['url']}")
    print("-" * 80)
