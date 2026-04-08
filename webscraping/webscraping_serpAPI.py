import pandas as pd
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
import os
import time
import dotenv

dotenv.load_dotenv()
from webscraping.qwen import QwenLLM

qwen = QwenLLM()

SERPAPI_KEY = "YOUR_SERPAPI_KEY"
INPUT_CSV = "dataset.csv"
OUTPUT_CSV = "dataset_summarized.csv"
CREDIT_LIMIT = 30000

def call_llm_for_summary(scraped_text):
    """Call local llm for summarization."""
    if not scraped_text or len(scraped_text) < 50:
        return "Not enough information to summarize."
    
    # Truncate text to avoid blowing up the LLM's context window
    max_chars = 6000 
    truncated_text = scraped_text[:max_chars]
    
    # return qwen.invoke(truncated_text)
    return f"SIMULATED SUMMARY OF: {truncated_text[:50]}..." 

def scrape_website(url):
    """Scrape the main text from a given URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract paragraph text for best descriptions
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        # Fallback if no <p> tags exist
        if len(text) < 50:
            text = soup.get_text(separator=' ', strip=True)
            
        return text
    except requests.exceptions.RequestException as e:
        print(f"  [!] Failed to scrape {url}: {e}")
        return None

def get_url_from_serpapi(business_name, address):
    """Use SerpAPI to find the best website link."""
    query = f'"{business_name}" {address}'
    params = {
      "engine": "google",
      "q": query,
      "api_key": SERPAPI_KEY,
      "num": 3 # Limit results payload to save bandwidth
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "organic_results" in results and len(results["organic_results"]) > 0:
            return results["organic_results"][0].get("link")
        return None
    except Exception as e:
        print(f"  [!] SerpAPI search failed: {e}")
        return None

def main():
    # Load dataset or resume from checkpoint
    if os.path.exists(OUTPUT_CSV):
        print(f"Loading progress from {OUTPUT_CSV}...")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        print(f"Loading raw data from {INPUT_CSV}...")
        df = pd.read_csv(INPUT_CSV)
    
    # Ensure 'summary' column exists
    if 'summary' not in df.columns:
        df['summary'] = None

    api_credits_used = 0
    save_interval = 50 

    for index, row in df.iterrows():
        # 1. CHECK IF SUMMARY IS MISSING
        if pd.notna(row['summary']) and str(row['summary']).strip() != "":
            continue # Skip if we already have a summary

        # 2. ENFORCE CREDIT LIMIT
        if api_credits_used >= CREDIT_LIMIT:
            print("\n[!] 30,000 API Credit Limit Reached. Halting to prevent overages.")
            break 

        business_name = row['Business_Name']
        address = row['FULL_ADDRESS']
        print(f"\nProcessing [{index}]: {business_name}")
        
        # 3. SEARCH VIA SERPAPI (Ignoring dataset link)
        found_url = get_url_from_serpapi(business_name, address)
        api_credits_used += 1
        
        scraped_text = None
        if found_url:
            print(f"  -> Found best link: {found_url}")
            scraped_text = scrape_website(found_url)
        else:
            print("  -> No results found on Google.")

        # 4. SUMMARIZE
        if scraped_text:
            summary = call_llm_for_summary(scraped_text)
            df.at[index, 'summary'] = summary
            print("  -> Summary generated.")
        else:
            df.at[index, 'summary'] = "Data unavailable for summarization."
            print("  -> Marked as unavailable.")

        # 5. CHECKPOINT SAVE
        if (index + 1) % save_interval == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"  [+] Progress saved. Credits used: {api_credits_used}")
            time.sleep(1) # Breather for LLM

    # Final save after finishing or hitting the API limit
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Total SerpAPI calls used this run: {api_credits_used}")
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()