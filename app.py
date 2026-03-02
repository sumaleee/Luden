import os
import json
import difflib
from flask import Flask, render_template, request
from build_ast import get_canonical_ast_signature

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(SCRIPT_DIR, "ast_database.json")

print("Loading Prefix AST Database...")
with open(DB_FILE, "r", encoding="utf-8") as f:
    ast_database = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    results_data = []
    
    if request.method == "POST":
        query = request.form.get("query", "")
        if query.strip():
            query_result = get_canonical_ast_signature(query)
            
            if query_result and query_result[0]:
                query_ast, query_vars = query_result
                candidates = []
                
                for db_item in ast_database:
                    # STRICT MATHEMATICAL BOUNDARY: Must have exact same number of variables
                    if db_item["num_vars"] != query_vars:
                        continue
                        
                    db_ast = db_item["ast_signature"]
                    structural_score = difflib.SequenceMatcher(None, query_ast, db_ast).ratio()
                    
                    candidates.append({
                        "id": db_item["id"],
                        "distance": round(structural_score, 4),
                        "problem_latex": db_item["raw_latex"],
                        "url": db_item["source_url"]
                    })
                
                candidates.sort(key=lambda x: x["distance"], reverse=True)
                results_data = candidates[:3]

    return render_template("index.html", query=query, results=results_data)

if __name__ == "__main__":
    print("\n--- ATTEMPTING TO START SERVER ON http://127.0.0.1:5000 ---")
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)