import os
import re
import json
import hashlib
import logging
from datetime import datetime
from sympy import expand, symbols, latex
from sympy.core import Symbol, Integer, Float, Rational
from sympy.core.traversal import preorder_traversal
from latex2sympy2_extended import latex2sympy
from openai import OpenAI

# --- CONFIGURATION ---
# Swap primary/fallback model here if uptime degrades
PRIMARY_MODEL   = "arcee-ai/trinity-large-preview:free"
FALLBACK_MODEL  = "arcee-ai/trinity-mini:free"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CACHE_FILE      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalizer_cache.json")
FAILURE_LOG     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalizer_failures.log")

# --- LOGGING (Failures only) ---
logging.basicConfig(
    filename=FAILURE_LOG,
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# --- CACHE ---
# Simple JSON cache: maps input string hash -> canonical LaTeX string.
# Prevents redundant LLM calls for repeated inputs at both index and query time.
def _load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def _cache_key(text):
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


# --- STAGE 1: LLM NORMALIZATION (Plain Text / Ambiguous Input -> LaTeX) ---
# Called when input is not already clean LaTeX (e.g. scraped plaintext, user typed input).
# Uses OpenRouter. Falls back to secondary model if primary fails.

_llm_client = None

def _get_client():
    global _llm_client
    if _llm_client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set. "
                "Export it in your shell: export OPENROUTER_API_KEY=your_key_here"
            )
        _llm_client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE
        )
    return _llm_client

SYSTEM_PROMPT = """You are a math notation converter.
Your only job is to convert the input math expression or equation into valid LaTeX.
Output ONLY the raw LaTeX string. No explanation. No preamble. No markdown code blocks.
Do not wrap the output in \\( \\) or $$ delimiters. Just the LaTeX expression itself.

Example:
Input: 5y - 3(4 - y) = 2y + 9
Output: 5y - 3(4 - y) = 2y + 9

Example:
Input: (4 - 2z)/3 = 3/4 - 5z/6
Output: \\frac{4 - 2z}{3} = \\frac{3}{4} - \\frac{5z}{6}

Example:
Input: 4t / (t^2 - 25) = 1 / (5 - t)
Output: \\frac{4t}{t^{2} - 25} = \\frac{1}{5 - t}"""

def _call_llm(text, model):
    """Make a single LLM call via OpenRouter. Returns raw response string."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text}
        ],
        temperature=0,      # Deterministic output - we want consistent LaTeX
        max_tokens=256      # Math expressions are short; cap to avoid runaway responses
    )
    return response.choices[0].message.content

def _strip_llm_response(raw):
    """
    Postprocessing safety net.
    Models sometimes wrap output in markdown or repeat the prompt format
    despite instructions. This strips all of that.
    """
    text = raw.strip()

    # Remove markdown code blocks: ```latex ... ``` or ``` ... ```
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = re.sub(r"```", "", text)

    # Remove "Output:" prefix if model mimics prompt format
    text = re.sub(r"(?i)^output\s*:\s*", "", text)

    # Remove LaTeX delimiters \( \) and $$ $
    text = re.sub(r"^\\\(|\\\)$", "", text).strip()
    text = re.sub(r"^\$\$|\$\$$", "", text).strip()
    text = re.sub(r"^\$|\$$", "", text).strip()

    return text.strip()

def to_latex(text):
    """
    Stage 1: Convert any math input format to a LaTeX string.
    Uses PRIMARY_MODEL with automatic fallback to FALLBACK_MODEL.
    Results are cached to avoid redundant API calls.
    """
    cache = _load_cache()
    key = _cache_key(text)

    if key in cache:
        return cache[key]

    raw = None
    model_used = None

    # Try primary model first
    try:
        raw = _call_llm(text, PRIMARY_MODEL)
        model_used = PRIMARY_MODEL
    except Exception as e:
        logging.warning(f"PRIMARY MODEL FAILED | model={PRIMARY_MODEL} | input={text!r} | error={e}")
        print(f"  [WARN] Primary model failed, trying fallback...")

        # Try fallback model
        try:
            raw = _call_llm(text, FALLBACK_MODEL)
            model_used = FALLBACK_MODEL
        except Exception as e2:
            logging.error(f"FALLBACK MODEL FAILED | model={FALLBACK_MODEL} | input={text!r} | error={e2}")
            print(f"  [ERROR] Both models failed for input: {text!r}")
            return text  # Last resort: return original input unchanged

    result = _strip_llm_response(raw)

    # Log if result looks suspicious (empty or unchanged from input)
    if not result or result == text:
        logging.warning(f"SUSPICIOUS OUTPUT | model={model_used} | input={text!r} | output={result!r}")

    # Cache and return
    cache[key] = result
    _save_cache(cache)
    return result


# --- STAGE 2: SYMPY CANONICALIZATION (LaTeX -> Canonical String) ---
# Parses LaTeX into a SymPy expression tree, expands it, abstracts
# all numeric coefficients, and re-serializes to a normalized LaTeX string.
# This is the mathematically principled replacement for normalize_aggressive().

def _abstract_numbers(expr):
    """
    Walk the SymPy expression tree and:
    1. Replace all numeric atoms (Integer, Float, Rational) with symbol N.
    2. Replace all variable symbols with canonical symbol x.
    This makes structurally identical equations with different variable
    names or coefficients produce identical canonical strings.
    Preserves mathematical constants (pi, E) which are not Symbol instances.
    """
    N = symbols('N')
    x = symbols('x')
    substitutions = {}
    for atom in preorder_traversal(expr):
        if isinstance(atom, (Integer, Float, Rational)):
            if atom != -1:
                substitutions[atom] = N
        elif isinstance(atom, Symbol) and atom.name not in ('N', 'x'):
            # Replace any variable (y, z, t, w, etc.) with x
            substitutions[atom] = x
    return expr.subs(substitutions)

def _parse_equation(latex_str):
    """
    Handle equations (with =) by splitting into LHS and RHS,
    parsing each side separately, then reassembling.
    Returns a tuple (lhs_expr, rhs_expr) or raises on failure.
    """
    # Split on = but not == and not \leq, \geq etc.
    sides = re.split(r'(?<![<>!\\])=(?!=)', latex_str, maxsplit=1)
    if len(sides) == 2:
        lhs = latex2sympy(sides[0].strip())
        rhs = latex2sympy(sides[1].strip())
        return lhs, rhs
    else:
        # No equals sign - treat as single expression
        return latex2sympy(latex_str), None

def canonicalize(latex_str):
    """
    Stage 2: Convert a LaTeX string to a canonical normalized string for embedding.

    Pipeline:
        LaTeX -> SymPy tree -> expand() -> abstract coefficients -> re-serialize

    Falls back to the regex normalizer if SymPy parsing fails.
    All failures are logged to normalizer_failures.log for later analysis.
    """
    try:
        lhs, rhs = _parse_equation(latex_str)

        # Expand both sides (distributes multiplication, removes factored forms)
        lhs_expanded = expand(lhs)

        if rhs is not None:
            rhs_expanded = expand(rhs)
            # Abstract numeric coefficients on both sides
            lhs_abstract = _abstract_numbers(lhs_expanded)
            rhs_abstract = _abstract_numbers(rhs_expanded)
            # Re-serialize to LaTeX string
            return latex(lhs_abstract) + " = " + latex(rhs_abstract)
        else:
            lhs_abstract = _abstract_numbers(lhs_expanded)
            return latex(lhs_abstract)

    except Exception as e:
        logging.warning(f"SYMPY PARSE FAILED | input={latex_str!r} | error={e} | falling back to regex")
        return _regex_fallback(latex_str)

def _regex_fallback(latex_str):
    """
    Fallback normalization using the original regex approach.
    Used when SymPy cannot parse the input.
    Existence of this fallback means the pipeline never hard-fails.
    """
    if not latex_str:
        return ""
    text = latex_str.lower()
    text = text.replace("show step", "")
    text = re.sub(r'(?<![a-z\\])[a-z](?![a-z])', 'x', text)
    text = re.sub(r'\d+(\.\d+)?', 'N', text)
    return text


# --- PUBLIC INTERFACE ---
# This is the single function called by build_db.py, interactive_tester.py,
# and test_suite.py. Input can be LaTeX, plain text, or anything in between.

def normalize(text, is_latex=False):
    """
    Full normalization pipeline. Converts any math input to a
    canonical string suitable for embedding.

    Args:
        text:      The input math expression (any format).
        is_latex:  Set True to skip the LLM stage (input is already clean LaTeX).
                   Use this for database entries scraped as LaTeX.

    Returns:
        A normalized canonical string for embedding.
    """
    if not text:
        return ""

    # Stage 1: Convert to LaTeX (skip if already LaTeX)
    if is_latex:
        latex_str = text
    else:
        latex_str = to_latex(text)

    # Stage 2: SymPy canonicalization
    return canonicalize(latex_str)


# --- QUICK SELF-TEST ---
# Run this file directly to verify the pipeline is working.
# python normalizer.py

if __name__ == "__main__":
    print("=" * 60)
    print("  NORMALIZER SELF-TEST")
    print("=" * 60)

    test_pairs = [
        # These two should produce identical canonical output
        {
            "label": "A (LaTeX, is_latex=True)",
            "input": "4x - 7(2 - x) = 3x + 2",
            "is_latex": True
        },
        {
            "label": "B (Plain text, is_latex=False)",
            "input": "5y - 3(4 - y) = 2y + 9",
            "is_latex": False
        },
    ]

    results = []
    for t in test_pairs:
        print(f"\nInput  [{t['label']}]: {t['input']}")
        result = normalize(t["input"], is_latex=t["is_latex"])
        print(f"Output: {result}")
        results.append(result)

    print("\n" + "-" * 60)
    if results[0] == results[1]:
        print("[PASS] Both inputs produced identical canonical output.")
    else:
        print("[INFO] Outputs differ — review above.")
        print(f"  A: {results[0]}")
        print(f"  B: {results[1]}")
    print("=" * 60)
