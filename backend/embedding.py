from sentence_transformers import SentenceTransformer
import json
import os

raw_path = "../data/raw/hitman_games.json"
processed_path = "../data/processed/hitman_games_with_embeddings.json"

with open(raw_path, "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer('all-mpnet-base-v2')

for item in data:
    text = item['text']
    embedding = model.encode(text, show_progress_bar=True)
    item['embedding'] = embedding.tolist()

os.makedirs(os.path.dirname(processed_path), exist_ok=True)
with open(processed_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Embedding completed and saved to {processed_path}")