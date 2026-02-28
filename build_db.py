import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
import re

# CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "linear_equations_data.json")
DB_PATH = os.path.join(SCRIPT_DIR, "math_knowledge_base")
COLLECTION_NAME = "algebra_problems"
MODEL_NAME = "all-mpnet-base-v2"

# --- AGGRESSIVE NORMALIZATION (regex) ---
def normalize_aggressive(latex_str):
    if not latex_str: return ""
    text = latex_str.lower()
    text = text.replace("show step", "")
    
    # 1. Variables -> 'x'
    text = re.sub(r'(?<![a-z\\])[a-z](?![a-z])', 'x', text)
    
    # 2. Numbers -> 'N'
    # This forces the AI to match structure, not values.
    text = re.sub(r'\d+(\.\d+)?', 'N', text)
    
    return text

print(f"Loading AI Model ({MODEL_NAME})...")
model = SentenceTransformer(MODEL_NAME)

print(f"Connecting to Database at {DB_PATH}...")
client = chromadb.PersistentClient(path=DB_PATH)

# RESET DATABASE (Critical to remove old 'MiniLM' or non-aggressive vectors)
try:
    client.delete_collection(name=COLLECTION_NAME)
    print("  (Deleted old collection to start fresh)")
except:
    pass

collection = client.create_collection(name=COLLECTION_NAME)

print(f"Loading data from {DATA_FILE}...")
with open(DATA_FILE, "r", encoding="utf-8") as f:
    problems = json.load(f)

ids = []
documents = []             # Display (Original)
normalized_documents = []  # Embed (Aggressive)
metadatas = []

print("Processing problems...")
for p in problems:
    ids.append(p["id"])
    documents.append(p["problem_latex"])
    
    # Apply the Aggressive Normalization
    norm_math = normalize_aggressive(p["problem_latex"])
    normalized_documents.append(norm_math)
    
    metadatas.append({
        "url": p["url"],
        "topic": p["topic"],
        "solution": p["solution_text"][:1000] 
    })

print(f"Indexing {len(documents)} items...")
embeddings = model.encode(normalized_documents).tolist()

collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)

print("\n[SUCCESS] Database Rebuilt with Aggressive Normalization!")