import chromadb
from sentence_transformers import SentenceTransformer
import sys
import os
from LLM_normalizer import normalize

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "math_knowledge_base")
COLLECTION_NAME = "algebra_problems"
MODEL_NAME = "all-mpnet-base-v2"

print(f"Loading Model: {MODEL_NAME}...")
try:
    model = SentenceTransformer(MODEL_NAME, device='cpu')

    print(f"Connecting to Database at {DB_PATH}...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    sys.exit()

print("\n" + "="*50)
print(f"   MATH SEARCH ENGINE ({collection.count()} problems)")
print("="*50)
print("Type a math problem (or 'exit' to quit).")

while True:
    query = input("\nYour Problem > ")
    if query.lower() in ['exit', 'quit']:
        break

    # Normalize user input — unknown format, run full pipeline (LLM + SymPy)
    canonical = normalize(query, is_latex=False)
    print(f"   (Canonical: {canonical})")

    # Search
    results = collection.query(
        query_embeddings=model.encode([canonical]).tolist(),
        n_results=1
    )

    if results['ids']:
        match_dist = results['distances'][0][0]
        match_tex = results['documents'][0][0]
        match_sol = results['metadatas'][0][0]['solution']

        print(f"\n[BEST MATCH FOUND]")
        print(f"Confidence Score: {1 - match_dist:.4f} (Distance: {match_dist:.4f})")
        print(f"Database Problem: {match_tex}")
        print("-" * 40)

        sol_preview = match_sol.replace("\n", " ")[:300]
        print(f"Solution Strategy: {sol_preview}...")
    else:
        print("No match found.")
