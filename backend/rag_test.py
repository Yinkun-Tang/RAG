import json
from hybrid_retriever import HybridRetriever
from google import genai

# Initialize Gemini client (reads GEMINI_API_KEY from environment)
client = genai.Client()

QUERY = "Why is Hitman: Blood Money popular?"
FAISS_TOP_K = 50
BM25_TOP_K = 50
FINAL_TOP_K = 5
MODEL = "gemini-2.5-flash"

retriever = HybridRetriever(
    faiss_index_path="../data/processed/hitman_faiss.index",
    metadata_path="../data/processed/hitman_index_mapping.json"
)

# Retrieve top documents
results = retriever.search(
    QUERY,
    faiss_top_k=FAISS_TOP_K,
    bm25_top_k=BM25_TOP_K,
    final_top_k=FINAL_TOP_K
)

# Build context string
context_texts = [
    f"[{r['page_title']} - {r['section']}] {r['text']}"
    for r in results
]
context = "\n\n".join(context_texts)

# Construct prompt
prompt = f"""
You are a Hitman series encyclopedia expert.
Use the following context to answer the question.
Cite sources using [Title - Section].

Context:
{context}

Question:
{QUERY}

Answer concisely with references.
"""

# Call Gemini model via SDK
response = client.models.generate_content(
    model=MODEL,
    contents=prompt
)

print("RAG answer:\n")
print(response.text)

print("\nReferences:")
for r in results:
    print(f"- {r['page_title']} [{r['section']}] - {r['url']}")
