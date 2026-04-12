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

# CONFIG
SERPAPI_KEY = os.getenv("SERP_API_KEY")
INPUT_FILE = "data/split_data_with_summaries/train_with_summaries.csv"
OUTPUT_FILE = "serp_dataset_summarized_train.csv"

MAX_CHARS = 6000
BATCH_SIZE = 15              # How many rows to process before saving a checkpoint
MAX_CONCURRENT_TABS = 5      # How many browser tabs to open at once
TIMEOUT = 15000              # 15 seconds for page loads
CREDIT_LIMIT = 5000

if not SERPAPI_KEY:
    print("\n[!] CRITICAL ERROR: SERP_API_KEY is missing. Check your .env file.")
    exit()

qwen = QwenLLM()

# Shared state to track API usage and early exits
state = {"credits_used": 0, "limit_reached": False}

async def async_get_urls_from_serpapi(business_name, address):
    """Asynchronously calls SerpAPI to find the top 5 website links and a fallback snippet."""
    if state["credits_used"] >= CREDIT_LIMIT:
        if not state["limit_reached"]:
            print(f"\n[!] LIMIT REACHED: {CREDIT_LIMIT} credits used. Flagging script to halt.")
            state["limit_reached"] = True
        return [], "" # Return empty list and empty snippet

    # PRE-EMPTIVELY claim the credit to prevent concurrent tasks from bypassing the limit
    state["credits_used"] += 1

    query = f'"{business_name}" {address}'
    url = f"https://serpapi.com/search.json?engine=google&q={query}&api_key={SERPAPI_KEY}&num=5" # Changed to 5
    
    # Stagger the API calls by 0.1 to 1.5 seconds to prevent 401 Rate Limits from concurrency
    await asyncio.sleep(random.uniform(0.1, 1.5))
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    urls = []
                    fallback_snippet = ""
                    
                    if "organic_results" in data and len(data["organic_results"]) > 0:
                        # Save the top result's snippet as our ultimate fallback
                        fallback_snippet = data["organic_results"][0].get("snippet", "")
                        
                        # Extract up to 5 URLs
                        for result in data["organic_results"][:5]: # Changed to 5
                            if "link" in result:
                                urls.append(result["link"])
                                
                    return urls, fallback_snippet
                else:
                    error_text = await response.text()
                    print(f"  [!] SerpAPI Error {response.status}: {error_text}")
                    state["credits_used"] -= 1 
                    return [], ""
                    
        return [], "" 
        
    except Exception as e:
        # REFUND the credit if the API call actually crashed/failed
        state["credits_used"] -= 1 
        print(f"  [!] SerpAPI request failed: {e}")
        return [], ""

async def get_content(context, url: str) -> str:
    """Scrapes paragraph text with retry logic and error handling."""
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
    """Sends text to local LLM via thread to prevent blocking the async event loop."""
    if not isinstance(text, str) or len(text.strip()) < 50:
        return "Insufficient content for summary."
    try:
        return await asyncio.to_thread(qwen.invoke, text)
    except Exception as e:
        return f"Summary failed: {str(e)}"

async def append_business_description_data():
    print("--- Starting Scraper and Summarizer ---")
    
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing output. Loading {OUTPUT_FILE}...")
        df = pd.read_csv(OUTPUT_FILE, dtype={'summary': object})
    else:
        if not os.path.exists(INPUT_FILE):
            print(f"Error: {INPUT_FILE} not found.")
            return
        print(f"Starting fresh from {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)
    
    if 'summary' not in df.columns:
        df['summary'] = None
    df['summary'] = df['summary'].astype(object)

    empty_mask = (
        df['summary'].isna() | 
        (df['summary'].astype(str).str.strip() == "") | 
        (df['summary'].astype(str).str.contains("Insufficient content for summary", case=False, na=False))
    )
    indices_to_process = df.index[empty_mask].tolist()
    total_to_do = len(indices_to_process)

    if total_to_do == 0:
        print("All items have been processed!")
        return

    print(f"Found {total_to_do} empty items. First target index: {indices_to_process[0]}")

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
                
                print(f"[{current_count}/{total}] Searching SerpAPI for: {business_name}")
                
                # Fetch up to 5 URLs and the #1 snippet
                urls, fallback_snippet = await async_get_urls_from_serpapi(business_name, address)
                
                if urls:
                    # Loop through the URLs one by one
                    for i, url in enumerate(urls):
                        url = url.strip().lower()
                        if not url.startswith(('http://', 'https://')):
                            url = "https://" + url
                            
                        print(f"[{current_count}/{total}] Scraping Attempt {i+1}/5: {url}")
                        scraped_text = await get_content(context, url)
                        
                        # If we get good text, stop searching and return it!
                        if scraped_text and len(scraped_text) > 50:
                            return scraped_text
                            
                        # If it failed, print a message and let the loop try the next URL
                        if i < len(urls) - 1:
                            print(f"  -> Scrape failed for {url}. Trying next link...")
                        else:
                            print(f"  -> All 5 links failed to scrape for {business_name}.")

                # THE ULTIMATE FALLBACK: If we have no URLs, or all 5 scrapes failed
                if fallback_snippet:
                    print(f"  -> Falling back to Google snippet for {business_name}.")
                    return fallback_snippet
                else:
                    return ""

        for i in range(0, total_to_do, BATCH_SIZE):
            batch_indices = indices_to_process[i : i + BATCH_SIZE]
            
            print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ({len(batch_indices)} items) ---")
            
            tasks = []
            for idx, real_idx in enumerate(batch_indices):
                row = df.loc[real_idx]
                tasks.append(
                    sem_task(
                        business_name=row['Business_Name'],
                        address=row['FULL_ADDRESS'],
                        current_count=i + idx + 1,
                        total=total_to_do
                    )
                )
            
            scraped_results = await asyncio.gather(*tasks)
            
            print(f"Summarizing batch with local LLM...")
            for idx, text in enumerate(scraped_results):
                real_idx = batch_indices[idx]
                
                # ONLY summarize and update if we actually got text back (or a snippet)
                if text and text.strip():
                    summary = await async_summarize(text)
                    df.loc[real_idx, 'summary'] = summary
                else:
                    print(f"  -> Skipped updating row {real_idx} (no text extracted or limit reached)")
            
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Checkpoint saved: {i + len(batch_indices)}/{total_to_do} items completed.")
            print(f"SerpAPI Credits Used So Far: {state['credits_used']}")

            # --- EARLY EXIT CHECK ---
            if state["limit_reached"]:
                print("\n[!] Halting script execution to protect API limit.")
                break

        await browser.close()

    print(f"\nTask Complete! Final data saved to {OUTPUT_FILE}")
    print(f"Total SerpAPI Credits used this session: {state['credits_used']}")

if __name__ == "__main__":
    # Playwright on Windows requires the default Proactor event loop to launch browsers
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
    asyncio.run(append_business_description_data())