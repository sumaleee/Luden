import requests
from bs4 import BeautifulSoup
import copy
import time
import json
import re

BASE_URL = "https://tutorial.math.lamar.edu"
INDEX_URL = f"{BASE_URL}/Classes/Alg/Alg.aspx"
OUTPUT_FILE = "algebra_data.json"

def extract_latex_from_element(el):
    tex = el.find("script", type="math/tex")
    if tex:
        return tex.get_text(strip=True)

    el_copy = copy.deepcopy(el)
    for a in el_copy.find_all("a"):
        a.decompose()
    raw = el_copy.get_text(" ", strip=True)

    m = re.search(r'\\\((.+?)\\\)', raw, re.DOTALL)
    if m:
        return "\\(" + m.group(1).strip() + "\\)"
        
    m2 = re.search(r'\\\[(.+?)\\\]', raw, re.DOTALL)
    if m2:
        return "\\[" + m2.group(1).strip() + "\\]"

    return None

def main():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0"
    })
    
    print(f"1. Harvesting topic slugs from {INDEX_URL}...")
    r = session.get(INDEX_URL, timeout=10)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    slugs = []
    for div in soup.find_all('div', class_='indent'):
        for a in div.find_all('a', class_='introlink', href=True):
            href = a['href']
            if href.endswith('.aspx'):
                slugs.append(href.replace('.aspx', ''))
                
    print(f"Found {len(slugs)} distinct topics.")
    print("-" * 50)
    
    dataset = []
    problem_count = 0
    
    for slug in slugs:
        # FACTUAL CORRECTION: The correct directories for Practice and Assignments
        target_urls = [
            f"{BASE_URL}/Problems/Alg/{slug}.aspx",
            f"{BASE_URL}/ProblemsNS/Alg/{slug}.aspx"
        ]
        
        for url in target_urls:
            print(f"Scraping: {url} ...", end=" ", flush=True)
            time.sleep(1) 
            
            try:
                r = session.get(url, timeout=10)
                if r.status_code == 404 or "Page Not Found" in r.text:
                    print("Skipped (No Page)")
                    continue
            except Exception as e:
                print(f"Failed ({e})")
                continue
                
            page_soup = BeautifulSoup(r.text, 'html.parser')
            problem_lists = page_soup.find_all('ol', class_=['practice-problems', 'assign-problems'])
            
            if not problem_lists:
                print("Skipped (No problem list tags)")
                continue
                
            found_here = 0
            for ol in problem_lists:
                for li in ol.find_all('li', recursive=False):
                    tex = extract_latex_from_element(li)
                    if tex:
                        problem_count += 1
                        found_here += 1
                        dataset.append({
                            "id": f"pauls_alg_{problem_count:04d}",
                            "source_url": url,
                            "problem_latex": tex
                        })
            
            print(f"Found {found_here} problems.")

    print("-" * 50)
    print(f"Extraction complete. Writing {problem_count} total problems to {OUTPUT_FILE}.")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

if __name__ == "__main__":
    main()