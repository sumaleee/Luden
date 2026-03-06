# Luden - Structural Math Problem Matcher

A Flask web app that takes a math equation entered via an interactive keyboard, parses it into an Abstract Syntax Tree (AST), and retrieves the most structurally identical problems from a curated dataset sourced from Paul's Online Math Notes.

---

## How It Works

```
User Input (MathLive keyboard)
    → LaTeX String
    → AST Normalization (latex2sympy2 + custom prefix serializer)
    → Hierarchical Similarity Scoring against AST Database
    → Top 10 Structural Matches (rendered with MathJax)
```

The core insight is **structural invariance**: `5x + 2 = 12` and `4y + 7 = 19` are the same *type* of problem. The AST parser replaces concrete numbers with `C` and variables with `X1, X2, …`, so matching is driven entirely by mathematical shape — not surface text.

---

## Tech Stack

| Component        | Library / Tool                          |
| ---------------- | --------------------------------------- |
| Language         | Python 3                                |
| Web Framework    | Flask                                   |
| Math Parsing     | latex2sympy2_extended + SymPy           |
| Similarity Score | Custom hierarchical scorer + difflib    |
| Math Input       | MathLive (browser virtual keyboard)     |
| Math Rendering   | MathJax 3                               |
| Data Scraping    | requests + BeautifulSoup4               |
| Data Source      | Paul's Online Math Notes (Algebra)      |

---

## Project Structure

```
Luden/
├── app.py               # Flask server — scoring logic and route handler
├── build_ast.py         # AST builder — parses LaTeX into prefix signatures,
│                        #   rebuilds ast_database.json
├── scraper.py           # Scrapes algebra problems from Paul's Online Notes
│                        #   → outputs algebra_data.json
├── algebra_data.json    # Raw scraped dataset (1,122 problems)
├── ast_database.json    # Parsed AST signature database (1,048 entries)
├── failed_parses.log    # Problems that failed AST parsing (for debugging)
├── templates/
│   └── index.html       # Frontend — MathLive keyboard, MathJax results,
│                        #   dark/light mode toggle
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/sumaleee/luden.git
cd Luden

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

---

## Usage

### 1. (Optional) Re-scrape the dataset

```bash
python scraper.py
```

Crawls Paul's Online Math Notes algebra section and writes `algebra_data.json`.

### 2. (Optional) Rebuild the AST database

```bash
python build_ast.py
```

Parses every problem in `algebra_data.json` into a normalized prefix AST signature and writes `ast_database.json`. Re-run this whenever the parser logic changes.

### 3. Run the web app

```bash
python app.py
```

Opens at `http://127.0.0.1:5000`. Enter an equation using the on-screen math keyboard and click **Find Matches**.

---

## Scoring

Matches are ranked by a weighted hierarchical score (0–1):

| Weight | Component                        | What it captures                                      |
| ------ | -------------------------------- | ----------------------------------------------------- |
| 45%    | Top-level operator match         | Same problem *type* (Equality, FiniteSet, Pow, …)     |
| 30%    | Structural operator set overlap  | Same *kinds* of operations (trig, log, sqrt, …)       |
| 25%    | Token sequence similarity        | Fine-grained shape similarity (via `difflib`)         |

Results are color-coded by score:

- **Green** (≥ 0.70) — strong structural match
- **Yellow** (0.45 – 0.69) — similar shape, some differences
- **Red** (< 0.45) — weak match

---

## AST Normalization

Key behaviors of the prefix serializer (`build_ast.py`):

- **Variables** → `X1, X2, …` (sorted alphabetically, remapped per problem)
- **Constants / Numbers** → `C`
- **Imaginary unit `i`** → `I` (treated as a named constant, never a variable)
- **Rational numbers** (e.g. `1/2`, `3/4`) → `Div C C`
- **Division** (SymPy `Mul` with `Pow(expr, -1)` args) → `Div`
- **Square root** (`Pow(expr, 1/2)` or `Pow(expr, Pow(2,-1))`) → `sqrt`
- **nth root** (`Pow(expr, 1/n)` for n > 2) → `nthroot`
- **Label-side stripping**: for equations like `y = 3x-1` or `f(x) = x²`, the lone variable / function-application side is stripped so matching focuses on the structurally rich side
- **Commutative ops** (`Add`, `Mul`) → args sorted, so `x+3` and `3+x` produce identical signatures

---

## Development Log

### Phase 1 — Data Acquisition
Scraped algebra problems from Paul's Online Math Notes across all topic pages. 1,122 problems harvested into `algebra_data.json`.

### Phase 2 — Vector Database (retired)
ChromaDB initialized with persistent storage; problems embedded with `all-mpnet-base-v2`. Retired in Phase 3 because semantic embeddings matched by topic keywords, not mathematical structure.

### Phase 3 — Structural Matching (current)
Four iterations to reach the current approach:

- **Iteration 1 — Base NLP:** Failed. The model matched rational equations with linear equations just because they shared the variable `y`.
- **Iteration 2 — Model Upgrade:** Swapped MiniLM for `all-mpnet-base-v2`. Improved, but still insufficient.
- **Iteration 3 — Aggressive Regex:** Replaced all variables with `x` and numbers with `N`. Worked for simple cases but failed on edge cases (couldn't distinguish Euler's `e` from a variable).
- **Iteration 4 — AST Parsing (current):** Replaced regex with `latex2sympy2_extended`. LaTeX is parsed into a SymPy expression tree, traversed in prefix order, and normalized. Ongoing edge-case fixes:
  - `sqrt` / `nthroot` separated from general `Pow` (handles both `Rational(1/n)` and `Pow(n,-1)` exponent forms)
  - Imaginary unit `i` treated as constant, not variable
  - Label-side stripping for single-variable equations

### Phase 4 — Web Interface
Flask app with MathLive input keyboard and MathJax rendering. Dark mode by default with light mode toggle. Top 10 matches shown with color-coded similarity scores.

### Phase 5 — Vision Layer (pending)
Integrate `pix2tex` to accept a screenshot from the clipboard and extract a LaTeX string automatically.

---

## Roadmap

- [ ] Integrate pix2tex for clipboard OCR
- [ ] Expand dataset to calculus topics (derivatives, integrals)
- [ ] Improve scoring for systems of equations and piecewise functions
