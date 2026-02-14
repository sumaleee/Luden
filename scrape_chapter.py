import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import time
import json
import random

# --- CONFIGURATION ---
BASE_URL = "https://tutorial.math.lamar.edu"
TARGET_URL = "https://tutorial.math.lamar.edu/Problems/Alg/SolveLinearEqns.aspx"
OUTPUT_FILE = "linear_equations_data.json"

# --- ENGINEERING: RETRY STRATEGY ---
# We create a session that automatically retries failed requests
session = requests.Session()
retry = Retry(
    total=3,                # Retry 3 times
    backoff_factor=1,       # Wait 1s, 2s, 4s between retries
    status_forcelist=[500, 502, 503, 504], # Retry on server errors
    allowed_methods=["HEAD", "GET", "OPTIONS"] # Retry only on safe methods
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Add headers to look like a real browser
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
})

def get_soup(url):
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status() # Raise error for bad status codes (404, 500)
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        print(f"    [ERROR] Failed to fetch {url}: {e}")
        return None

# --- MAIN EXECUTION ---
print(f"Starting scrape of: {TARGET_URL}")
soup = get_soup(TARGET_URL)

if not soup:
    print("Critical Error: Could not load index page. Exiting.")
    exit()

# Find the list of problems
problem_list = soup.find("ol", class_="practice-problems")
if not problem_list:
    print("Could not find problem list (<ol class='practice-problems'>).")
    exit()

items = problem_list.find_all("li")
print(f"Found {len(items)} problems. Beginning harvest...")

database = []

for index, item in enumerate(items):
    # 1. Extract Problem
    tex_script = item.find("script", type="math/tex")
    if tex_script:
        problem_tex = tex_script.get_text(strip=True)
    else:
        problem_tex = item.get_text(" ", strip=True).split("Solution")[0].strip()

    # 2. Extract Solution Link
    sol_link = item.find("a", class_="practice-soln-link")
    
    if sol_link:
        rel_url = sol_link['href']
        full_sol_url = BASE_URL + rel_url
        
        print(f"  [{index+1}/{len(items)}] Scraping solution...")
        
        # VISIT SOLUTION PAGE
        sol_soup = get_soup(full_sol_url)
        
        if sol_soup:
            # Using the class we discovered earlier: 'soln-content'
            sol_div = sol_soup.find("div", class_="soln-content")
            
            if sol_div:
                solution_text = sol_div.get_text(" ", strip=True)
                
                # Save Data
                entry = {
                    "id": f"alg_linear_{index}",
                    "url": full_sol_url,
                    "problem_latex": problem_tex,
                    "solution_text": solution_text,
                    "topic": "Linear Equations"
                }
                database.append(entry)
            else:
                print(f"    [WARNING] No 'soln-content' div found.")
        
        # Polite Delay
        time.sleep(random.uniform(1.0, 2.5))
        
    else:
        print(f"  [SKIP] No solution link for problem {index+1}")

# --- SAVE TO FILE ---
print(f"\nFinished. Saving {len(database)} items to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(database, f, indent=4)

print("Done.")