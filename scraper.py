import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import copy
import time
import json
import random
import re
import os

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CHAPTERS = [
    ("SolveLinearEqns",       "Linear Equations"),
    ("SolveQuadraticEqnsI",   "Quadratic Equations I"),
    ("SolveQuadraticEqnsII",  "Quadratic Equations II"),
    ("ReducibleToQuadratic",  "Reducible to Quadratic"),
    ("SolveRadicalEqns",      "Radical Equations"),
    ("SolveAbsValueEqns",     "Absolute Value Equations"),
]

BASE_URL       = "https://tutorial.math.lamar.edu"
PROBLEMS_BASE  = f"{BASE_URL}/Problems/Alg"
SOLUTIONS_BASE = f"{BASE_URL}/Solutions/Alg"
OUTPUT_FILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algebra_data.json")

# ---------------------------------------------------------------------------
# SESSION
# ---------------------------------------------------------------------------
session = requests.Session()
retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://",  adapter)
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})

def get_soup(url):
    try:
        r = session.get(url, timeout=15)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser")
    except Exception as e:
        print(f"    [ERROR] {url}: {e}")
        return None

def polite_delay():
    time.sleep(random.uniform(1.5, 3.0))

# ---------------------------------------------------------------------------
# EXTRACT LATEX FROM AN ELEMENT'S TEXT
# ---------------------------------------------------------------------------
def extract_latex_from_element(el):
    # Method 1: legacy <script type="math/tex"> tag
    tex = el.find("script", type="math/tex")
    if tex:
        return tex.get_text(strip=True)

    # Method 2: \(...\) inline in raw text — strip anchor tags first
    el_copy = copy.deepcopy(el)
    for a in el_copy.find_all("a"):
        a.decompose()
    raw = el_copy.get_text(" ", strip=True)

    m = re.search(r'\\\((.+?)\\\)', raw, re.DOTALL)
    if m:
        return "\\(" + m.group(1).strip() + "\\)"

    return None

# ---------------------------------------------------------------------------
# EXTRACT PROBLEM LATEX FROM A SOLUTION PAGE
# ---------------------------------------------------------------------------
def extract_latex_from_solution_page(sol_soup, prob_num):
    # Regex to capture either inline \( ... \) or display \[ ... \] math
    pattern = r'(\\\(.*?\\\)|\\\[.*?\\\])'
    
    # Strategy A: Look in known problem statement CSS classes
    for class_name in ["problem-statement", "practice-problem", "problem", "prob-stmt"]:
        div = sol_soup.find("div", class_=class_name)
        if div:
            m = re.search(pattern, div.get_text(" ", strip=True))
            if m:
                return m.group(1).strip()

    # Strategy B: DOM traversal - look at elements immediately preceding the solution content
    sol_div = sol_soup.find("div", class_="soln-content")
    if sol_div:
        # Check direct previous siblings
        for prev in sol_div.find_previous_siblings():
            m = re.search(pattern, prev.get_text(" ", strip=True))
            if m:
                return m.group(1).strip()
        
        # Check parent's previous siblings if deeply nested
        parent = sol_div.parent
        if parent:
            for prev in parent.find_previous_siblings():
                m = re.search(pattern, prev.get_text(" ", strip=True))
                if m:
                    return m.group(1).strip()

    # Strategy C: Structural Heuristic - find the first math expression containing an '=' sign
    page_text = sol_soup.get_text(" ", strip=True)
    for m in re.finditer(pattern, page_text):
        expr = m.group(1).strip()
        if '=' in expr:
            return expr

    # Strategy D: Absolute fallback - the very first math expression on the page
    m = re.search(pattern, page_text)
    if m:
        return m.group(1).strip()

    return None

# ---------------------------------------------------------------------------
# SCRAPE ONE CHAPTER
# ---------------------------------------------------------------------------
def scrape_chapter(slug, topic, chapter_index):
    index_url = f"{PROBLEMS_BASE}/{slug}.aspx"
    print(f"\n{'='*60}")
    print(f"  Chapter : {topic}")
    print(f"  URL     : {index_url}")
    print(f"{'='*60}")

    soup = get_soup(index_url)
    if not soup:
        print("  [SKIP] Could not load index page.")
        return []

    # Update to search for either list class
    problem_list = soup.find("ol", class_=["practice-problems", "assign-problems"])
    if not problem_list:
        print("  [SKIP] No problem list found.")
        return []

    # --- Pass 1: collect problems visible in static HTML ---
    scraped = {}  # prob_number -> {problem_latex, sol_url}

    sol_links = problem_list.find_all("a", class_="practice-soln-link")
    print(f"  Static HTML shows {len(sol_links)} solution links.")

    for sol_link in sol_links:
        item = sol_link.find_parent("li")
        if not item:
            continue
            
        href = sol_link.get("href", "")
        num_match = re.search(r'Prob(\d+)\.aspx', href, re.IGNORECASE)
        if not num_match:
            continue
        prob_num = int(num_match.group(1))

        problem_tex = extract_latex_from_element(item)
        scraped[prob_num] = {
            "problem_latex": problem_tex,
            "sol_url": BASE_URL + href,
        }

    # --- Pass 2: probe for problems beyond what static HTML shows ---
    max_seen = max(scraped.keys()) if scraped else 0
    probe_num = max_seen + 1
    consecutive_misses = 0

    print(f"  Probing for problems beyond Prob{max_seen}...")
    while consecutive_misses < 2:
        probe_url = f"{SOLUTIONS_BASE}/{slug}/Prob{probe_num}.aspx"
        try:
            r = session.get(probe_url, timeout=10, allow_redirects=False)
            
            if r.status_code == 200 and "soln-content" in r.text:
                consecutive_misses = 0
                if probe_num not in scraped:
                    scraped[probe_num] = {
                        "problem_latex": None,
                        "sol_url": probe_url,
                    }
                    print(f"  [PROBE] Found Prob{probe_num}")
            else:
                consecutive_misses += 1
        except Exception as e:
            print(f"  [PROBE] Error on Prob{probe_num}: {e}")
            consecutive_misses += 1
            
        probe_num += 1
        time.sleep(0.5)

    print(f"  Total problems to process: {len(scraped)}")

    # --- Fetch solution pages and build final entries ---
    results = []

    for prob_num in sorted(scraped.keys()):
        data = scraped[prob_num]
        sol_url = data["sol_url"]
        problem_tex = data["problem_latex"]

        print(f"  [Prob {prob_num:02d}] Fetching solution page...")
        sol_soup = get_soup(sol_url)
        if not sol_soup:
            polite_delay()
            continue

        sol_div = sol_soup.find("div", class_="soln-content")
        if not sol_div:
            print(f"    [WARN] No soln-content div found.")
            polite_delay()
            continue
        solution_text = sol_div.get_text(" ", strip=True)

        if not problem_tex:
            problem_tex = extract_latex_from_solution_page(sol_soup, prob_num)

        if not problem_tex:
            print(f"    [WARN] Could not find problem LaTeX — skipping Prob{prob_num}.")
            polite_delay()
            continue

        results.append({
            "id": f"alg_{chapter_index}_{prob_num}",
            "url": sol_url,
            "problem_latex": problem_tex,
            "solution_text": solution_text,
            "topic": topic,
            "chapter": slug,
        })
        print(f"         latex : {problem_tex[:70]}")
        polite_delay()

    return results

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
all_problems = []

for chapter_index, (slug, topic) in enumerate(CHAPTERS):
    chapter_problems = scrape_chapter(slug, topic, chapter_index)
    all_problems.extend(chapter_problems)
    print(f"\n  -> {len(chapter_problems)} problems collected from {topic}.")
    time.sleep(random.uniform(2.0, 4.0))

print(f"\n{'='*60}")
print(f"  TOTAL: {len(all_problems)} problems.")
print(f"  Saving to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_problems, f, indent=4, ensure_ascii=False)
print("  Done.")