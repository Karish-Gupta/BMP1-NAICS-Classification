import asyncio
import aiohttp
import pandas as pd
import os
import re
import random
import dotenv
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from webscraping.qwen import QwenLLM

dotenv.load_dotenv()

# --- CONFIGURATION ---
SERPAPI_KEY = os.getenv("SERP_API_KEY")
INPUT_FILE = "data/split_data_with_summaries/test_with_summaries.csv"
OUTPUT_FILE = "serp_dataset_summarized_test.csv"

START_INDEX = 1000           # Only process rows starting from this index
CREDIT_LIMIT = 500          # Total credits to use for this session
MAX_CHARS = 6000
BATCH_SIZE = 15              # Rows per checkpoint save
MAX_CONCURRENT_TABS = 5      # Simultaneous browser tabs
TIMEOUT = 15000              # 15 seconds for page loads

if not SERPAPI_KEY:
    print("\n[!] CRITICAL ERROR: SERP_API_KEY is missing. Check your .env file.")
    exit()

# Initialize local LLM
qwen = QwenLLM()

# Shared state to track API usage and early exits
state = {"credits_used": 0, "limit_reached": False}

async def async_get_urls_from_serpapi(business_name, address):
    """Asynchronously calls SerpAPI with 401 auto-retry logic and top 5 links."""
    if state["credits_used"] >= CREDIT_LIMIT:
        if not state["limit_reached"]:
            print(f"\n[!] LIMIT REACHED: {CREDIT_LIMIT} credits used. Flagging script to halt.")
            state["limit_reached"] = True
        return [], ""

    # Pre-emptively claim the credit
    state["credits_used"] += 1
    query = f'"{business_name}" {address}'
    url = f"https://serpapi.com/search.json?engine=google&q={query}&api_key={SERPAPI_KEY}&num=5"
    
    for attempt in range(3):
        # Stagger requests to avoid clumping and 401 errors
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        urls = []
                        fallback_snippet = ""
                        
                        if "organic_results" in data and len(data["organic_results"]) > 0:
                            # Use the top result's snippet as the ultimate fallback
                            fallback_snippet = data["organic_results"][0].get("snippet", "")
                            
                            for result in data["organic_results"][:5]:
                                if "link" in result:
                                    urls.append(result["link"])
                                    
                        return urls, fallback_snippet
                        
                    elif response.status == 401:
                        print(f"  [!] SerpAPI 401 Rate Limit for '{business_name}'. Retrying ({attempt+1}/3)...")
                        await asyncio.sleep(3)
                        continue 
                        
                    else:
                        error_text = await response.text()
                        print(f"  [!] SerpAPI Error {response.status}: {error_text}")
                        state["credits_used"] -= 1 
                        return [], ""
                        
        except Exception as e:
            if attempt == 2:
                state["credits_used"] -= 1 
                print(f"  [!] SerpAPI request failed completely: {e}")
                return [], ""
            
    state["credits_used"] -= 1 
    return [], ""

async def get_content(context, url: str) -> str:
    """Scrapes paragraph text with retry logic."""
    for attempt in range(2):
        page = await context.new_page()
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")
            await asyncio.sleep(1) 

            elements = await page.locator('p').all()
            paragraphs = []
            for p in elements:
                try:
                    txt = await p.inner_text(timeout=2000)
                    clean_txt = txt.strip().replace('\n', ' ')
                    if len(clean_txt) > 50:
                        paragraphs.append(clean_txt)
                except:
                    continue
            
            content = " ".join(paragraphs).strip()
            if not content or len(content) < 100:
                body_text = await page.inner_text('body', timeout=5000)
                content = re.sub(r'\s+', ' ', body_text).strip()

            if content:
                return content[:MAX_CHARS]
            
        except (PlaywrightTimeout, Exception) as e:
            if attempt == 0:
                url = url.replace("https://", "http://")
            else:
                print(f"  [!] Failed to scrape {url} | Reason: {type(e).__name__}")
        finally:
            await page.close()
    return ""

async def async_summarize(text):
    """Sends text to local LLM."""
    if not isinstance(text, str) or len(text.strip()) < 50:
        return "Insufficient content for summary."
    try:
        return await asyncio.to_thread(qwen.invoke, text)
    except Exception as e:
        return f"Summary failed: {str(e)}"

async def append_business_description_data():
    print(f"--- Starting Scraper (START_INDEX: {START_INDEX}) ---")
    
    if os.path.exists(OUTPUT_FILE):
        print(f"Loading existing output: {OUTPUT_FILE}")
        df = pd.read_csv(OUTPUT_FILE, dtype={'summary': object})
    else:
        print(f"Starting fresh from: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
    
    if 'summary' not in df.columns:
        df['summary'] = None
    df['summary'] = df['summary'].astype(object)

    # Filtering Logic: Missing, Empty, or previously failed summaries + START_INDEX
    empty_mask = (
        df['summary'].isna() | 
        (df['summary'].astype(str).str.strip() == "") | 
        (df['summary'].astype(str).str.contains("Insufficient content for summary", case=False, na=False))
    )
    indices_to_process = df.index[empty_mask & (df.index >= START_INDEX)].tolist()
    total_to_do = len(indices_to_process)

    if total_to_do == 0:
        print("No rows meet the criteria for processing.")
        return

    print(f"Found {total_to_do} target rows. First index: {indices_to_process[0]}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080}
        )
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TABS)

        async def sem_task(business_name, address, current_count, total):
            async with semaphore:
                if state["limit_reached"]:
                    return ""
                
                print(f"[{current_count}/{total}] Searching SerpAPI: {business_name}")
                urls, fallback_snippet = await async_get_urls_from_serpapi(business_name, address)
                
                if urls:
                    for i, url in enumerate(urls):
                        url_clean = url.strip().lower()
                        if not url_clean.startswith(('http://', 'https://')):
                            url_clean = "https://" + url_clean
                            
                        print(f"[{current_count}/{total}] Scraping Attempt {i+1}/5: {url_clean}")
                        scraped_text = await get_content(context, url_clean)
                        
                        if scraped_text and len(scraped_text) > 50:
                            return scraped_text
                            
                        if i < len(urls) - 1:
                            print(f"  -> Scrape failed for {url_clean}. Trying next...")

                if fallback_snippet:
                    print(f"  -> Falling back to Google snippet for {business_name}.")
                    return fallback_snippet
                return ""

        for i in range(0, total_to_do, BATCH_SIZE):
            batch_indices = indices_to_process[i : i + BATCH_SIZE]
            print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ({len(batch_indices)} items) ---")
            
            tasks = [sem_task(df.loc[idx, 'Business_Name'], df.loc[idx, 'FULL_ADDRESS'], i + j + 1, total_to_do) 
                     for j, idx in enumerate(batch_indices)]
            
            scraped_results = await asyncio.gather(*tasks)
            
            print(f"Summarizing batch...")
            for idx, text in enumerate(scraped_results):
                real_idx = batch_indices[idx]
                if text and text.strip():
                    summary = await async_summarize(text)
                    df.loc[real_idx, 'summary'] = summary
                else:
                    print(f"  -> Skipped row {real_idx} (No data found)")
            
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Checkpoint saved. SerpAPI Credits Used: {state['credits_used']}")

            if state["limit_reached"]:
                break

        await browser.close()

    print(f"\nTask Complete! Total credits used: {state['credits_used']}")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(append_business_description_data())