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
    text = re.sub(r'(?<![a-z\\])[a-z](?![a-z])', 'x', text) # Var -> x
    text = re.sub(r'\d+(\.\d+)?', 'N', text)                # Num -> N
    return text

# Load Resources
print("Loading Model...")
model = SentenceTransformer(MODEL_NAME, device='cpu')
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# --- THE TARGETED TEST CASES ---
# We create a "Mutation" of each of your 6 scraped problems.
test_cases = [
    {
        "name": "Prob 1 Variant (Linear Dist)",
        # Original: 4x - 7(2 - x) = 3x + 2
        "query": "5y - 3(4 - y) = 2y + 9", 
        "expected_id": "alg_linear_0" 
    },
    {
        "name": "Prob 2 Variant (Double Dist)",
        # Original: 2(w + 3) - 10 = 6(32 - 3w)
        "query": "3(z + 5) - 4 = 2(10 - 2z)", 
        "expected_id": "alg_linear_1"
    },
    {
        "name": "Prob 3 Variant (Rational Simple)",
        # Original: (4 - 2z)/3 = 3/4 - 5z/6
        "query": "(2 - 3x)/5 = 1/2 - 4x/3",
        "expected_id": "alg_linear_2"
    },
    {
        "name": "Prob 4 Variant (Rational Diff Squares)",
        # Original: 4t / (t^2 - 25) = 1 / (5 - t)
        "query": "3x / (x^2 - 16) = 1 / (4 - x)", 
        "expected_id": "alg_linear_3"
    },
    {
        "name": "Prob 5 Variant (Rational Linear Denom)",
        # Original: (3y + 4) / (y - 1) = 2 + 7 / (y - 1)
        "query": "(2x + 5) / (x - 3) = 4 + 1 / (x - 3)",
        "expected_id": "alg_linear_4"
    },
    {
        "name": "Prob 6 Variant (Rational Complex)",
        # Original: 5x / (3x - 3) + 6 / (x + 2) = 5/3
        "query": "2x / (4x - 4) + 3 / (x + 1) = 7/2", 
        "expected_id": "alg_linear_5"
    },
    {
        "name": "Distractor (Quadratic)",
        "query": "x^2 + 5x + 6 = 0",
        "expected_id": "NONE"
    }
]

print("\n" + "="*95)
print(f"{'TEST NAME':<30} | {'DIST':<6} | {'STATUS':<10} | {'MATCH PREVIEW'}")
print("="*95)

for test in test_cases:
    clean_query = normalize_aggressive(test["query"])
    
    results = collection.query(
        query_embeddings=model.encode([clean_query]).tolist(),
        n_results=1
    )
    
    if results['ids']:
        dist = results['distances'][0][0]
        found_id = results['ids'][0][0]
        match_tex = results['documents'][0][0]
        
        # Determine Pass/Fail
        # We consider it a PASS if:
        # 1. The ID matches the expected ID
        # 2. OR if it's a Distractor and the distance is high (> 0.4)
        
        status = "FAIL"
        if test["expected_id"] == "NONE":
            if dist > 0.4: status = "PASS"
            else: status = "FAIL (False Positive)"
        elif found_id == test["expected_id"]:
            if dist < 0.2: status = "PASS"
            else: status = "WEAK MATCH"
        else:
            status = f"FAIL (Got {found_id})"

        # Truncate preview
        preview = (match_tex[:40] + '..') if len(match_tex) > 40 else match_tex
        
        print(f"{test['name']:<30} | {dist:.4f} | {status:<10} | {preview}")
    else:
        print(f"{test['name']:<30} | N/A    | ERROR      | No Match")

print("="*95 + "\n")