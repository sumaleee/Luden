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

# Load Resources
print("Loading Model...")
model = SentenceTransformer(MODEL_NAME, device='cpu')
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# --- TEST CASES ---
# Each query is a structural variant of a database problem.
# Different variable names and coefficients, same algebraic structure.
test_cases = [
    {
        "name": "Prob 1 Variant (Linear Dist)",
        # Original: 4x - 7(2 - x) = 3x + 2
        "query": "5y - 3(4 - y) = 2y + 9",
        "expected_id": "alg_0_1"
    },
    {
        "name": "Prob 2 Variant (Double Dist)",
        # Original: 2(w + 3) - 10 = 6(32 - 3w)
        "query": "3(z + 5) - 4 = 2(10 - 2z)",
        "expected_id": "alg_0_2"
    },
    {
        "name": "Prob 3 Variant (Rational Simple)",
        # Original: (4 - 2z)/3 = 3/4 - 5z/6
        "query": "(2 - 3x)/5 = 1/2 - 4x/3",
        "expected_id": "alg_0_3"
    },
    {
        "name": "Prob 4 Variant (Rational Diff Squares)",
        # Original: 4t / (t^2 - 25) = 1 / (5 - t)
        "query": "3x / (x^2 - 16) = 1 / (4 - x)",
        "expected_id": "alg_0_4"
    },
    {
        "name": "New: Quadratic Variant",
        # Original (alg_1_1): u^2 - 5u - 14 = 0
        "query": "x^2 - 7x - 18 = 0",
        "expected_id": "alg_1_1"
    },
    {
        "name": "New: Absolute Value Variant",
        # Original (alg_5_1): |4p - 7| = 3
        "query": "|2x - 5| = 8",
        "expected_id": "alg_5_1"
    },
    {
        "name": "Distractor (Cubic - Should Fail)",
        "query": "x^3 + 5x^2 + 6x = 0",
        "expected_id": "NONE"
    }
]

print("\n" + "="*105)
print(f"{'TEST NAME':<35} | {'DIST':<6} | {'STATUS':<22} | {'CANONICAL':<25} | {'MATCH PREVIEW'}")
print("="*105)

passes = 0
fails = 0

for test in test_cases:
    # Full pipeline — test queries are plain text / unknown format
    canonical = normalize(test["query"], is_latex=False)

    results = collection.query(
        query_embeddings=model.encode([canonical]).tolist(),
        n_results=1
    )

    if results['ids']:
        dist = results['distances'][0][0]
        found_id = results['ids'][0][0]
        match_tex = results['documents'][0][0]

        status = "FAIL"
        if test["expected_id"] == "NONE":
            # If we expect no match, the distance should be relatively high
            if dist > 0.2:
                status = "PASS"
                passes += 1
            else:
                status = "FAIL (False Positive)"
                fails += 1
        elif found_id == test["expected_id"]:
            if dist < 0.2:
                status = "PASS"
                passes += 1
            else:
                status = "WEAK MATCH"
                fails += 1
        else:
            status = f"FAIL (Got {found_id})"
            fails += 1

        preview = (match_tex[:30] + '..') if len(match_tex) > 30 else match_tex
        canonical_preview = (canonical[:25]) if len(canonical) > 25 else canonical

        print(f"{test['name']:<35} | {dist:.4f} | {status:<22} | {canonical_preview:<25} | {preview}")
    else:
        print(f"{test['name']:<35} | N/A    | ERROR                 | {canonical[:25]:<25} | No Match")
        fails += 1

print("="*105)
print(f"\nResult: {passes}/{passes+fails} passed\n")