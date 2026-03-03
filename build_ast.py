import json
import re
from sympy.core.parameters import evaluate
from sympy import Tuple
from latex2sympy2_extended import latex2sympy

DATA_FILE = "algebra_data.json"
DB_FILE = "ast_database.json"
LOG_FILE = "failed_parses.log"

def clean_latex(latex_str):
    s = latex_str.strip()
    s = re.sub(r'^\\\(|\\\)$', '', s).strip()
    s = re.sub(r'^\\\[|\\\]$', '', s).strip()
    
    # Remove purely visual formatting
    s = s.replace(r'\displaystyle', '')
    s = s.replace(r'\left', '')
    s = s.replace(r'\right', '')
    s = s.replace(r'\,', '')  
    
    # Normalize \frac shorthand (MathLive omits braces for single chars)
    # \frac12 -> \frac{1}{2},  \frac{1}2 -> \frac{1}{2},  \frac1{2} -> \frac{1}{2}
    s = re.sub(r'\\frac([^{\\])([^{\\])', r'\\frac{\1}{\2}', s)
    s = re.sub(r'\\frac(\{[^{}]*\})([^{\\])', r'\\frac\1{\2}', s)
    s = re.sub(r'\\frac([^{\\])(\{)', r'\\frac{\1}\2', s)

    # Fix specific LaTeX syntax quirks that break the parser
    s = s.replace(r'\centerdot', r'\cdot')
    s = s.replace(r'\bf{e}', 'e')
    s = s.replace(r'{\bf{e}}', 'e')
    
    # NOTE: The destructive brace replacement regexes were intentionally removed 
    # from here to prevent nested fractions and exponents from unbalancing.
    return s.strip()

def parse_to_ast(cleaned_latex):
    """Custom parser to handle standard LaTeX and Systems of Equations."""
    # Intercept and structurally map Systems of Equations
    if r'\begin{align*}' in cleaned_latex:
        match = re.search(r'\\begin\{align\*\}(.*?)\\end\{align\*\}', cleaned_latex, re.DOTALL)
        if match:
            contents = match.group(1)
            # Split equations by newline
            eqs = contents.split(r'\\')
            parsed_eqs = []
            for eq in eqs:
                # Remove alignment ampersands
                eq_clean = eq.replace('&', '').strip()
                if eq_clean:
                    parsed_eqs.append(latex2sympy(eq_clean))
            # Return them as a grouped Tuple node in the AST
            if parsed_eqs:
                return Tuple(*parsed_eqs)
    
    # Normal single equation parsing
    return latex2sympy(cleaned_latex)

def to_prefix(node, var_map):
    if getattr(node, 'is_Number', False):
        if getattr(node, 'is_Rational', False) and not getattr(node, 'is_Integer', False):
            # Rational atom (e.g. 1/2, 3/4) has no args in SymPy, so expand it
            # explicitly as division: numerator C / denominator C
            return "Div C C"
        return "C"
    if node in var_map:
        return var_map[node]
    if not node.args:
        return str(node.func.__name__)

    func_name = node.func.__name__

    # Detect a/b: Mul where some args are Pow(expr, -1) — the SymPy representation
    # of division. Separating these out prevents rational expressions from being
    # indistinguishable from plain multiplication.
    if func_name == 'Mul':
        numer_args, denom_args = [], []
        for arg in node.args:
            if arg.func.__name__ == 'Pow' and len(arg.args) == 2 and arg.args[1] == -1:
                denom_args.append(arg.args[0])
            else:
                numer_args.append(arg)
        if denom_args:
            if not numer_args:
                numer_str = "C"
            elif len(numer_args) == 1:
                numer_str = to_prefix(numer_args[0], var_map)
            else:
                numer_str = "Mul " + " ".join(to_prefix(a, var_map) for a in numer_args)
            if len(denom_args) == 1:
                denom_str = to_prefix(denom_args[0], var_map)
            else:
                denom_str = "Mul " + " ".join(to_prefix(d, var_map) for d in denom_args)
            return f"Div {numer_str} {denom_str}"

    args_prefix = " ".join(to_prefix(arg, var_map) for arg in node.args)
    return f"{func_name} {args_prefix}"

def get_canonical_ast_signature(latex_str):
    cleaned = clean_latex(latex_str)
    try:
        with evaluate(False):
            expr = parse_to_ast(cleaned)
            
            free_syms = sorted(list(expr.free_symbols), key=lambda var: var.name)
            var_map = {sym: f"X{i+1}" for i, sym in enumerate(free_syms)}
            
            prefix_str = to_prefix(expr, var_map)
            return prefix_str, len(free_syms), None
            
    except Exception as e:
        return None, 0, str(e)

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        problems = json.load(f)
        
    print(f"Parsing {len(problems)} equations into Abstract Syntax Trees...")
    
    ast_database = []
    success_count = 0
    
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("--- AST PARSING FAILURES ---\n\n")
        
        for p in problems:
            raw_latex = p["problem_latex"]
            ast_signature, num_vars, error_msg = get_canonical_ast_signature(raw_latex)
            
            if ast_signature:
                success_count += 1
                ast_database.append({
                    "id": p["id"],
                    "raw_latex": raw_latex,
                    "source_url": p["source_url"],
                    "ast_signature": ast_signature,
                    "num_vars": num_vars
                })
            else:
                log.write(f"ID: {p['id']}\n")
                log.write(f"RAW LATEX: {raw_latex}\n")
                log.write(f"ERROR: {error_msg}\n")
                log.write("-" * 50 + "\n")
            
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(ast_database, f, indent=4)
        
    print(f"\nAST Database built. Successfully mapped {success_count}/{len(problems)} structures.")

if __name__ == "__main__":
    main()