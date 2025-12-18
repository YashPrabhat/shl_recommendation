import requests
from bs4 import BeautifulSoup
import json
import time
import re
import concurrent.futures
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION ---
BASE_URL = "https://www.shl.com/products/product-catalog/"
OUTPUT_FILE = "shl_assessments.json"

# Headers to look like a real browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Type=1 is "Individual Test Solutions"
PARAMS_TYPE = 1 
PAGE_SIZE = 12

def get_session():
    """Creates a request session with built-in retry logic."""
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def scrape_catalog_page(start, session):
    """Scrapes the main list page to get URLs and basic metadata."""
    params = {
        "start": start,
        "type": PARAMS_TYPE
    }
    
    # Retry loop specifically for ReadTimeout
    for attempt in range(3):
        try:
            # Increased timeout to 30 seconds
            r = session.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            break # Success, exit loop
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt+1}/3 failed for start={start}: {e}")
            time.sleep(5) # Wait 5 seconds before retrying
            if attempt == 2: # Last attempt failed
                return []

    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("tr[data-entity-id]")
    
    page_results = []
    
    for row in rows:
        # 1. Name & URL
        name_tag = row.select_one("td.custom__table-heading__title a")
        if not name_tag:
            continue
            
        name = name_tag.get_text(strip=True)
        href = name_tag["href"]
        url = "https://www.shl.com" + href if href.startswith("/") else href

        # 2. Test Types Mapping
        type_map = {
            "A": "Ability & Aptitude",
            "B": "Biodata & Situational Judgement",
            "C": "Competencies",
            "D": "Development & 360",
            "E": "Assessment Exercises",
            "K": "Knowledge & Skills",
            "P": "Personality & Behavior",
            "S": "Simulations"
        }
        
        found_types = []
        keys = row.select("span.product-catalogue__key")
        for k in keys:
            letter = k.get_text(strip=True)
            if letter in type_map:
                found_types.append(type_map[letter])
        
        if not found_types:
            found_types = ["General Assessment"]

        page_results.append({
            "name": name,
            "url": url,
            "test_type": list(set(found_types)),
            "remote_support": "Yes",
            "adaptive_support": "No"
        })
        
    return page_results

def scrape_details(assessment):
    """Visits the individual product page to get Description and Duration."""
    url = assessment['url']
    try:
        # Create a new session for thread safety or just use requests directly
        r = requests.get(url, headers=HEADERS, timeout=20)
        
        if r.status_code != 200:
            return assessment

        soup = BeautifulSoup(r.text, "html.parser")
        text_content = soup.get_text(" ", strip=True)

        # 1. Description
        desc = ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            desc = meta_desc.get("content", "").strip()
        
        if len(desc) < 10:
            main_div = soup.select_one("div.product-layout__content")
            if main_div:
                desc = main_div.get_text(strip=True)[:800]
            else:
                desc = text_content[:500]

        assessment['description'] = desc

        # 2. Duration (Regex)
        duration_match = re.search(r'(?:Time|Duration)[\s\w]*?(\d+)\s*(?:min|minute)', text_content, re.IGNORECASE)
        if duration_match:
            assessment['duration'] = int(duration_match.group(1))
        else:
            assessment['duration'] = 0

        # 3. Adaptive
        if "adaptive" in text_content.lower():
            assessment['adaptive_support'] = "Yes"

        print(f"Scraped details: {assessment['name'][:30]}...")
        return assessment

    except Exception as e:
        print(f"Error details for {url}: {e}")
        assessment['description'] = "No description available."
        assessment['duration'] = 0
        return assessment

def main():
    session = get_session()
    print("--- Stage 1: Collecting List of Assessments ---")
    all_assessments = []
    start = 0
    empty_pages = 0
    
    while True:
        print(f"Fetching page starting at {start}...")
        batch = scrape_catalog_page(start, session)
        
        if not batch:
            empty_pages += 1
            if empty_pages > 2: 
                break
        else:
            empty_pages = 0
            all_assessments.extend(batch)
        
        start += PAGE_SIZE
        if start > 1200: break # Safety limit
        time.sleep(0.5)

    # Deduplicate
    unique_map = {item['url']: item for item in all_assessments}
    final_list = list(unique_map.values())
    
    print(f"\nFound {len(final_list)} unique assessments. Minimum required: 377")
    
    print("\n--- Stage 2: Fetching Details (Description & Duration) ---")
    
    detailed_results = []
    
    # Increase workers slightly to speed it up, but not too much to cause timeouts
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_url = {executor.submit(scrape_details, item): item for item in final_list}
        
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                data = future.result()
                detailed_results.append(data)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    # Check count
    count = len(detailed_results)
    print(f"\nTotal Detailed Assessments: {count}")
    
    if count < 377:
        print("WARNING: Still under 377. Check manual scraping for missing pages.")
    
    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=4)
    
    print(f"SUCCESS: Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()