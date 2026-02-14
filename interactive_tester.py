import chromadb
from sentence_transformers import SentenceTransformer
import sys
import re
import os

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "math_knowledge_base")
COLLECTION_NAME = "algebra_problems"
MODEL_NAME = "all-mpnet-base-v2"

# --- AGGRESSIVE NORMALIZATION ---
def normalize_aggressive(latex_str):
    if not latex_str: return ""
    text = latex_str.lower()
    text = text.replace("show step", "")
    
    # 1. Variables -> 'x'
    text = re.sub(r'(?<![a-z\\])[a-z](?![a-z])', 'x', text)
    
    # 2. Numbers -> 'N'
    text = re.sub(r'\d+(\.\d+)?', 'N', text)
    
    return text

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
    
    # 1. Normalize User Input
    clean_query = normalize_aggressive(query)
    # print(f"   (Debug: Searching for '{clean_query}')") 
    
    # 2. Search
    results = collection.query(
        query_embeddings=model.encode([clean_query]).tolist(),
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