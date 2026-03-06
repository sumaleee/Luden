import json
import os
import re
from sympy.core.parameters import evaluate
from sympy import Tuple
from latex2sympy2_extended import latex2sympy

_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(_DIR, "algebra_data.json")
DB_FILE   = os.path.join(_DIR, "ast_database.json")
LOG_FILE  = os.path.join(_DIR, "failed_parses.log")

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

def _root_degree(exp):
    """Return integer n if exp represents 1/n (a root exponent), else None.

    Handles both SymPy forms:
      - Rational(1, n)  — produced when evaluate=True or by some parsers
      - Pow(n, -1)      — produced when evaluate=False keeps it unevaluated
    """
    # Rational(1, n) form
    if (getattr(exp, 'is_Rational', False)
            and not getattr(exp, 'is_Integer', False)
            and getattr(exp, 'p', None) == 1):
        return getattr(exp, 'q', None)
    # Pow(n, -1) form
    if exp.func.__name__ == 'Pow' and len(exp.args) == 2:
        b, e = exp.args
        if (getattr(b, 'is_Integer', False)
                and getattr(b, 'is_positive', False)
                and e == -1):
            return int(b)
    return None


def to_prefix(node, var_map):
    if getattr(node, 'is_Number', False):
        if getattr(node, 'is_Rational', False) and not getattr(node, 'is_Integer', False):
            # Rational atom (e.g. 1/2, 3/4) has no args in SymPy, so expand it
            # explicitly as division: numerator C / denominator C
            return "Div C C"
        return "C"
    # Imaginary unit: latex2sympy yields Symbol('i'), SymPy core yields ImaginaryUnit.
    # Either way, treat as a named constant so it never maps to X1/X2/…
    if node.func.__name__ == 'ImaginaryUnit':
        return "I"
    if node.is_Symbol and getattr(node, 'name', None) == 'i':
        return "I"
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
                parts = sorted(to_prefix(a, var_map) for a in numer_args)
                numer_str = "Mul " + " ".join(parts)
            if len(denom_args) == 1:
                denom_str = to_prefix(denom_args[0], var_map)
            else:
                parts = sorted(to_prefix(d, var_map) for d in denom_args)
                denom_str = "Mul " + " ".join(parts)
            return f"Div {numer_str} {denom_str}"

    # Distinguish sqrt / nthroot from general Pow so they don't cross-match.
    # SymPy can represent 1/n exponents two ways depending on evaluate context:
    #   Rational(1, n)  — e.g. sqrt(x) -> Pow(x, Rational(1,2))
    #   Pow(n, -1)      — e.g. cbrt(x^2) -> Pow(Pow(x,2), Pow(3,-1))
    if func_name == 'Pow' and len(node.args) == 2:
        base, exp = node.args
        n = _root_degree(exp)
        if n == 2:
            return f"sqrt {to_prefix(base, var_map)}"
        if n is not None and n > 2:
            return f"nthroot {to_prefix(base, var_map)} C"

    # Commutative ops: sort arg prefixes so x+3 and 3+x produce the same signature
    if func_name in ('Add', 'Mul'):
        arg_prefixes = sorted(to_prefix(arg, var_map) for arg in node.args)
        return f"{func_name} {' '.join(arg_prefixes)}"

    args_prefix = " ".join(to_prefix(arg, var_map) for arg in node.args)
    return f"{func_name} {args_prefix}"

def _is_label_side(node):
    """True if this side of an equation is just a variable label with no
    structural content — i.e. it shouldn't drive similarity matching.

    Covers:
      - bare Symbol:            y, A, f
      - applied undefined fn:   f(x), g(x, y)  — the function name carries no
                                structure beyond the variables it wraps
    """
    if node.is_Symbol:
        return True
    if (node.args
            and all(getattr(a, 'is_Symbol', False) for a in node.args)
            and type(node.func).__name__ == 'UndefinedFunction'):
        return True
    return False


def get_canonical_ast_signature(latex_str):
    cleaned = clean_latex(latex_str)
    try:
        with evaluate(False):
            expr = parse_to_ast(cleaned)

            # For equations like  y = 3x-1 / A = Pe^{rt} / f(x) = x^2,
            # one side is just a label — strip it so matching focuses on
            # the structurally rich side only.
            if expr.func.__name__ == 'Equality' and len(expr.args) == 2:
                lhs, rhs = expr.args
                if _is_label_side(lhs):
                    expr = rhs
                elif _is_label_side(rhs):
                    expr = lhs

            # Exclude 'i' — it's the imaginary unit, not a problem variable
            free_syms = sorted(
                [s for s in expr.free_symbols if getattr(s, 'name', None) != 'i'],
                key=lambda var: var.name
            )
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