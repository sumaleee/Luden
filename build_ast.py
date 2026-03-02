import json
import re
from sympy.core.parameters import evaluate
from latex2sympy2_extended import latex2sympy

DATA_FILE = "algebra_data.json"
DB_FILE = "ast_database.json"

def clean_latex(latex_str):
    s = latex_str.strip()
    s = re.sub(r'^\\\(|\\\)$', '', s).strip()
    s = re.sub(r'^\\\[|\\\]$', '', s).strip()
    s = s.replace(r'\displaystyle', '')
    s = s.replace(r'\left', '')
    s = s.replace(r'\right', '')
    s = s.replace(r'\,', '')  
    s = re.sub(r'\{\{+', '{', s)
    s = re.sub(r'\}\}+', '}', s)
    return s.strip()

def to_prefix(node, var_map):
    """Recursively converts an AST node into pure mathematical Prefix Notation."""
    if getattr(node, 'is_Number', False):
        return "C"
    if node in var_map:
        return var_map[node]
    if not node.args:
        return str(node.func.__name__)
    
    args_prefix = " ".join(to_prefix(arg, var_map) for arg in node.args)
    return f"{node.func.__name__} {args_prefix}"

def get_canonical_ast_signature(latex_str):
    cleaned = clean_latex(latex_str)
    try:
        with evaluate(False):
            expr = latex2sympy(cleaned)
            
            # Extract variables and map deterministically
            free_syms = sorted(list(expr.free_symbols), key=lambda var: var.name)
            var_map = {sym: f"X{i+1}" for i, sym in enumerate(free_syms)}
            
            # Generate pure prefix notation
            prefix_str = to_prefix(expr, var_map)
            return prefix_str, len(free_syms)
            
    except Exception:
        return None, 0

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        problems = json.load(f)
        
    print(f"Parsing {len(problems)} equations into Prefix ASTs...")
    
    ast_database = []
    success_count = 0
    
    for p in problems:
        raw_latex = p["problem_latex"]
        ast_result = get_canonical_ast_signature(raw_latex)
        
        if ast_result and ast_result[0]:
            ast_signature, num_vars = ast_result
            success_count += 1
            ast_database.append({
                "id": p["id"],
                "raw_latex": raw_latex,
                "source_url": p["source_url"],
                "ast_signature": ast_signature,
                "num_vars": num_vars
            })
            
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(ast_database, f, indent=4)
        
    print(f"AST Database built. Successfully mapped {success_count}/{len(problems)} structures.")

if __name__ == "__main__":
    main()