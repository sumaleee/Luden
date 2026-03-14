import os
import json
import difflib
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
from build_ast import get_canonical_ast_signature

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(SCRIPT_DIR, "ast_database.json")

print("Loading Prefix AST Database...")
with open(DB_FILE, "r", encoding="utf-8") as f:
    ast_database = json.load(f)

# Structural operators that define the "type" of a problem.
# Matching these should count for more than surface-level string similarity.
_STRUCTURAL_OPS = {
    'Equality', 'FiniteSet', 'Tuple',           # top-level forms
    'Div', 'Pow',                                # rational / power structure
    'Abs', 'sin', 'cos', 'tan', 'sec', 'csc',   # trig / special
    'cot', 'asin', 'acos', 'atan',
    'log', 'ln', 'exp',                          # exponential / log
    'sqrt', 'nthroot', 'Piecewise',
    'I',                                         # imaginary unit
}

def hierarchical_score(query_ast, db_ast):
    q = query_ast.split()
    d = db_ast.split()

    # 1. Top-level operator match (Equality, FiniteSet, …) — highest weight
    top_match = float(bool(q) and bool(d) and q[0] == d[0])

    # 2. Structural operator overlap — which "kinds" of operations appear
    q_ops = set(q) & _STRUCTURAL_OPS
    d_ops = set(d) & _STRUCTURAL_OPS
    union = q_ops | d_ops
    op_match = len(q_ops & d_ops) / len(union) if union else 1.0

    # 3. Fine-grained token sequence similarity
    seq_match = difflib.SequenceMatcher(None, query_ast, db_ast).ratio()

    return 0.45 * top_match + 0.30 * op_match + 0.25 * seq_match

_ocr_model = None

def get_ocr_model():
    global _ocr_model
    if _ocr_model is None:
        from pix2tex.cli import LatexOCR
        _ocr_model = LatexOCR()
    return _ocr_model

@app.route("/ocr", methods=["POST"])
def ocr():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    try:
        latex = get_ocr_model()(img)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"latex": latex})

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results_data = []
    query_ast = ""

    if request.method == "POST":
        query = request.form.get("query", "")
        if query.strip():
            print(f"[DEBUG] Received query: {repr(query)}")
            query_result = get_canonical_ast_signature(query)
            print(f"[DEBUG] AST result: {query_result}")

            # Unpack all 3 values to prevent a ValueError
            if query_result and query_result[0]:
                query_ast, query_vars, error_msg = query_result
                candidates = []

                for db_item in ast_database:
                    if db_item.get("num_vars") != query_vars:
                        continue

                    db_ast = db_item["ast_signature"]
                    score = hierarchical_score(query_ast, db_ast)

                    candidates.append({
                        "id": db_item["id"],
                        "distance": round(score, 4),
                        "problem_latex": db_item["raw_latex"],
                        "url": db_item["source_url"],
                        "ast_signature": db_ast
                    })

                candidates.sort(key=lambda x: x["distance"], reverse=True)
                results_data = candidates[:10]

    return render_template("index.html", query=query, results=results_data, query_ast=query_ast)



if __name__ == "__main__":
    print("\n--- ATTEMPTING TO START SERVER ON http://127.0.0.1:5000 ---")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)