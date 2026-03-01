import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
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
# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "algebra_data.json") 
DB_PATH = os.path.join(SCRIPT_DIR, "math_knowledge_base")
COLLECTION_NAME = "algebra_problems"
MODEL_NAME = "all-mpnet-base-v2"

print(f"Loading AI Model ({MODEL_NAME})...")
model = SentenceTransformer(MODEL_NAME)

print(f"Connecting to Database at {DB_PATH}...")
client = chromadb.PersistentClient(path=DB_PATH)

# RESET DATABASE
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
documents = []            # Display (Original LaTeX)
canonical_documents = []  # Embed (Canonical normalized form)
metadatas = []

print("Processing problems...")
for p in problems:
    ids.append(p["id"])
    documents.append(p["problem_latex"])

    # Database entries are already LaTeX — clean scraped formatting, then skip LLM stage
    canonical = normalize(clean_scraped_latex(p["problem_latex"]), is_latex=True)
    canonical_documents.append(canonical)
    print(f"  [{p['id']}] {p['problem_latex'][:50]}...")
    print(f"          -> {canonical}")

    metadatas.append({
        "url": p["url"],
        "topic": p["topic"],
        "solution": p["solution_text"][:1000]
    })

print(f"\nIndexing {len(documents)} items...")
embeddings = model.encode(canonical_documents).tolist()

collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)

print("\n[SUCCESS] Database rebuilt with SymPy canonical normalization!")
