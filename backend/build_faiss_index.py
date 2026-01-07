import json
import numpy as np
import faiss

processed_path = "../data/processed/hitman_games_with_embeddings.json"
index_path = "../data/processed/hitman_faiss.index"

with open(processed_path, "r", encoding="utf-8") as f:
    data = json.load(f)

embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors of dimension {dimension}")

faiss.write_index(index, index_path)
print(f"FAISS index saved to {index_path}")

mapping_path = "../data/processed/hitman_index_mapping.json"
mapping = [{"page_title": item["page_title"], "section": item["section"], "text": item["text"], "url": item["url"]} for item in data]
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print(f"Mapping saved to {mapping_path}")