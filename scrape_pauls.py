import requests
from bs4 import BeautifulSoup

# URLs
base_url = "https://tutorial.math.lamar.edu"
start_url = "https://tutorial.math.lamar.edu/Problems/Alg/SolveLinearEqns.aspx"

print(f"Step 1: Fetching Index Page...")
response = requests.get(start_url)
soup = BeautifulSoup(response.content, 'html.parser')

# 1. Find the list of problems
problem_list = soup.find("ol", class_="practice-problems")
if not problem_list:
    print("Error: Could not find problem list.")
    exit()

# 2. Get the first list item
first_item = problem_list.find("li")

# --- IMPROVED PROBLEM EXTRACTION ---
# Strategy: Try to find the MathJax script first. 
# If that fails, just grab the raw text of the list item (it usually contains the math).
tex_script = first_item.find("script", type="math/tex")

if tex_script:
    problem_text = tex_script.get_text(strip=True)
    print(f"\n[FOUND] Problem (LaTeX): {problem_text}")
else:
    # Fallback: Just get the text of the whole list item, ignoring the "Solution" link text
    # This is "dirty" but works if the script tag is missing in raw HTML
    problem_text = first_item.get_text(" ", strip=True).replace("Solution", "").strip()
    print(f"\n[FOUND] Problem (Raw Text): {problem_text}")

# 3. Get the Solution Link
sol_link = first_item.find("a", class_="practice-soln-link")

if sol_link:
    full_sol_url = base_url + sol_link['href']
    print(f"[FOUND] Solution URL: {full_sol_url}")
    
    # --- STEP 2: JUMP TO THE SOLUTION PAGE ---
    print(f"\nStep 2: Visiting Solution Page...")
    sol_response = requests.get(full_sol_url)
    sol_soup = BeautifulSoup(sol_response.content, 'html.parser')
    
    # --- THE FIX: Use 'soln-content' ---
    solution_div = sol_soup.find("div", class_="soln-content")
    
    if solution_div:
        print("\n--- SOLUTION TEXT ---")
        # Print the first 500 characters to verify we got it
        print(solution_div.get_text(" ", strip=True)[:500])
        print("\n[SUCCESS] We have a valid Data Point!")
    else:
        print("Error: Still could not find solution div with class 'soln-content'")

else:
    print("Error: No solution link found.")