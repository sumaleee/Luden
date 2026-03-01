import chromadb
from sentence_transformers import SentenceTransformer
import os
import json
import re
from LLM_normalizer import normalize

def clean_scraped_latex(latex_str):
    """
    Strip display formatting from scraped LaTeX so it matches
    the clean LaTeX format produced by canonicalize() on query inputs.
    """
    s = latex_str.strip()
    # Remove \( \) and \[ \] delimiters
    s = re.sub(r'^\\\(|\\\)$', '', s).strip()
    s = re.sub(r'^\\\[|\\\]$', '', s).strip()
    # Remove \displaystyle
    s = s.replace(r'\displaystyle', '').strip()
    # Remove \left and \right
    s = s.replace(r'\left', '').replace(r'\right', '')
    # Collapse double braces {{ }} -> { }
    s = re.sub(r'\{\{', '{', s)
    s = re.sub(r'\}\}', '}', s)
    # Collapse multiple spaces
    s = re.sub(r' +', ' ', s).strip()
    return s

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "math_knowledge_base")
DATA_FILE = os.path.join(SCRIPT_DIR, "algebra_data.json")
COLLECTION_NAME = "algebra_problems"
MODEL_NAME = "all-mpnet-base-v2"

print("Loading Model & Database...")
model = SentenceTransformer(MODEL_NAME, device='cpu')
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    problems = json.load(f)

print(f"\nLoaded {len(problems)} problems. Starting mass evaluation...\n")

passes = 0
collisions = 0
fails = 0

for p in problems:
    expected_id = p["id"]
    original_latex = p["problem_latex"]
    
    # Apply the same sanitization used during database construction
    cleaned_latex = clean_scraped_latex(original_latex)
    canonical = normalize(cleaned_latex, is_latex=True)
    
    results = collection.query(
        query_embeddings=model.encode([canonical]).tolist(),
        n_results=1
    )
    
    if results['ids']:
        dist = results['distances'][0][0]
        found_id = results['ids'][0][0]
        
        if found_id == expected_id:
            if dist < 0.15:
                passes += 1
            else:
                fails += 1
                print(f"[FAIL] {expected_id} matched itself but with high distance: {dist:.4f}")
        else:
            # It matched a different problem in the database.
            if dist < 0.15:
                collisions += 1
                print(f"[COLLISION] {expected_id} structurally collided with {found_id} (Dist: {dist:.4f})")
            else:
                fails += 1
                print(f"[FAIL] {expected_id} incorrectly matched {found_id} (Dist: {dist:.4f})")
    else:
        fails += 1
        print(f"[ERROR] {expected_id} returned no results.")

print("\n" + "="*60)
print("MASS TEST RESULTS")
print("="*60)
print(f"Exact Matches : {passes}")
print(f"Collisions    : {collisions} (Valid structural overlaps)")
print(f"Failures      : {fails} (Mathematical parsing or distance errors)")
print(f"Total Tested  : {len(problems)}")
print("="*60)