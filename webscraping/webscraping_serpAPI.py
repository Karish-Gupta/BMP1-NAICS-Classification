import asyncio
import aiohttp
import pandas as pd
import os
import re
import dotenv
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from webscraping.qwen import QwenLLM

dotenv.load_dotenv()


# CONFIG
SERPAPI_KEY = os.getenv("SERP_API_KEY")
INPUT_FILE = "data/split_data_with_summaries/train_with_summaries.csv"
OUTPUT_FILE = "serp_dataset_summarized.csv"

MAX_CHARS = 6000
BATCH_SIZE = 15              # How many rows to process before saving a checkpoint
MAX_CONCURRENT_TABS = 5      # How many browser tabs to open at once
TIMEOUT = 15000              # 15 seconds for page loads
CREDIT_LIMIT = 30000

qwen = QwenLLM()

def call_llm_for_summary(scraped_text):
    """
    Call local llm for summarization
    """
    # Simulated response
    if not scraped_text or len(scraped_text) < 50:
        return "Not enough information to summarize."
    
    return qwen.invoke(scraped_text)

# Shared state to track API usage safely across async tasks
state = {"credits_used": 0}

async def async_get_url_from_serpapi(business_name, address):
    """Asynchronously calls SerpAPI to find the missing website link."""
    if state["credits_used"] >= CREDIT_LIMIT:
        print(f"  [!] SerpAPI limit ({CREDIT_LIMIT}) reached. Skipping search for {business_name}.")
        return None

    query = f'"{business_name}" {address}'
    url = f"https://serpapi.com/search.json?engine=google&q={query}&api_key={SERPAPI_KEY}&num=3"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    state["credits_used"] += 1
                    data = await response.json()
                    if "organic_results" in data and len(data["organic_results"]) > 0:
                        return data["organic_results"][0].get("link")
        return None
    except Exception as e:
        print(f"  [!] SerpAPI request failed: {e}")
        return None

async def get_content(context, url: str) -> str:
    """Scrapes paragraph text with retry logic and error handling."""
    # Try twice: once with https, once with http if it fails
    for attempt in range(2):
        page = await context.new_page()
        try:
            await page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")
            await asyncio.sleep(1) # Wait for a moment to let dynamic scripts run

            # 1. Try to get all <p> tags (the best source for business descriptions)
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
            
            # 2. Join paragraphs
            content = " ".join(paragraphs).strip()

            # 3. Fallback: If no <p> tags found, grab the body text
            if not content or len(content) < 100:
                body_text = await page.inner_text('body', timeout=5000)
                content = re.sub(r'\s+', ' ', body_text).strip()

            if content:
                return content[:MAX_CHARS]
            
        except (PlaywrightTimeout, Exception) as e:
            if attempt == 0:
                # If https failed, try http
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
        # to_thread is crucial here so your local LLM doesn't freeze the scraping queue
        return await asyncio.to_thread(qwen.invoke, text)
    except Exception as e:
        return f"Summary failed: {str(e)}"

async def append_business_description_data():
    print("--- Starting Scraper and Summarizer ---")
    
    # 1. LOAD DATA
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing output. Loading {OUTPUT_FILE}...")
        df = pd.read_csv(OUTPUT_FILE, dtype={'summary': object})
    else:
        if not os.path.exists(INPUT_FILE):
            print(f"Error: {INPUT_FILE} not found.")
            return
        print(f"Starting fresh from {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)
    
    # Ensure summary column exists
    if 'summary' not in df.columns:
        df['summary'] = None
    df['summary'] = df['summary'].astype(object)

    # 2. FIND STARTING INDICES (Only target empty summaries)
    empty_mask = (df['summary'].isna() | (df['summary'].astype(str).str.strip() == ""))
    indices_to_process = df.index[empty_mask].tolist()
    total_to_do = len(indices_to_process)

    if total_to_do == 0:
        print("All items have been processed!")
        return

    print(f"Found {total_to_do} empty items. First target index: {indices_to_process[0]}")

    # 3. RUN ASYNC SCRAPING & SUMMARIZING
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080}
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TABS)

        async def sem_task(business_name, address, url, current_count, total):
            async with semaphore:
                # Clean URL / Check for NaN
                is_valid_url = isinstance(url, str) and url.strip() != "" and str(url).lower() != "nan"
                
                # If no valid URL in dataset, hit SerpAPI
                if not is_valid_url:
                    print(f"[{current_count}/{total}] No URL. Searching SerpAPI for: {business_name}")
                    url = await async_get_url_from_serpapi(business_name, address)
                
                # Format URL if we found/have one
                if url:
                    url = url.strip().lower()
                    if not url.startswith(('http://', 'https://')):
                        url = "https://" + url
                        
                    print(f"[{current_count}/{total}] Scraping: {url}")
                    return await get_content(context, url)
                else:
                    return ""

        # 4. BATCH PROCESSING LOOP
        for i in range(0, total_to_do, BATCH_SIZE):
            batch_indices = indices_to_process[i : i + BATCH_SIZE]
            
            print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ({len(batch_indices)} items) ---")
            
            # Create concurrent tasks for scraping/searching
            tasks = []
            for idx, real_idx in enumerate(batch_indices):
                row = df.loc[real_idx]
                tasks.append(
                    sem_task(
                        business_name=row['Business_Name'],
                        address=row['FULL_ADDRESS'],
                        url=row['Insured Website'],
                        current_count=i + idx + 1,
                        total=total_to_do
                    )
                )
            
            # Execute batch scraping
            scraped_results = await asyncio.gather(*tasks)
            
            # Summarize Batch and update DataFrame
            print(f"Summarizing batch with local LLM...")
            for idx, text in enumerate(scraped_results):
                real_idx = batch_indices[idx]
                summary = await async_summarize(text)
                df.loc[real_idx, 'summary'] = summary
            
            # Overwrite the CSV
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Checkpoint saved: {i + len(batch_indices)}/{total_to_do} items completed.")
            print(f"SerpAPI Credits Used So Far: {state['credits_used']}")

        await browser.close()

    print(f"\nTask Complete! Final data saved to {OUTPUT_FILE}")
    print(f"Total SerpAPI Credits used this session: {state['credits_used']}")

if __name__ == "__main__":
    # Windows-specific fix for asyncio event loops (prevents "Event loop is closed" errors)
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(append_business_description_data())