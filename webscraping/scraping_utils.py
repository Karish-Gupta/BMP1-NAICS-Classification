from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright
from webscraping.qwen import QwenLLM
import pandas as pd
import asyncio
import re
import os

# Global Constants
MAX_CHARS = 2500         # Max characters to save per site
TIMEOUT = 25000          # 25 seconds
MAX_CONCURRENT_TABS = 5 
BATCH_SIZE = 20 

# Initialize LLM
qwen = QwenLLM()

async def get_content(context, url: str) -> str:
    """Scrapes paragraph text with retry logic and error handling."""
    if not isinstance(url, str) or not url.strip():
        return ""
    
    # Clean URL
    url = url.strip().lower()
    if not url.startswith(('http://', 'https://')):
        url = "https://" + url

    # Try twice: once with https, once with http if it fails
    for attempt in range(2):
        page = await context.new_page()
        try:
            # Set a common User-Agent via context (already done in main)
            await page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")
            
            # Wait for a moment to let dynamic scripts run
            await asyncio.sleep(1)

            # 1. Try to get all <p> tags (the best source for business descriptions)
            elements = await page.locator('p').all()
            paragraphs = []
            for p in elements:
                try:
                    txt = await p.inner_text(timeout=2000)
                    clean_txt = txt.strip().replace('\n', ' ')
                    # Only keep strings that look like actual sentences
                    if len(clean_txt) > 50:
                        paragraphs.append(clean_txt)
                except:
                    continue
            
            # 2. Join paragraphs
            content = " ".join(paragraphs).strip()

            # 3. Fallback: If no <p> tags found, grab the body text
            if not content or len(content) < 100:
                body_text = await page.inner_text('body', timeout=5000)
                # Simple regex to remove extra whitespace/newlines
                content = re.sub(r'\s+', ' ', body_text).strip()

            if content:
                return content[:MAX_CHARS]
            
        except (PlaywrightTimeout, Exception) as e:
            if attempt == 0:
                # If https failed, try http
                url = url.replace("https://", "http://")
            else:
                print(f"Failed: {url} | Reason: {type(e).__name__}")
        finally:
            await page.close()
            
    return ""


async def async_summarize(text):
    if not isinstance(text, str) or len(text.strip()) < 50:
        return "Insufficient content for summary."
    try:
        return await asyncio.to_thread(qwen.invoke, text)
    except Exception as e:
        return f"Summary failed: {str(e)}"


async def append_business_description_data(INPUT_FILE, OUTPUT_FILE):
    print("--- Starting Scraper and Summarizer ---")
    
    # LOAD DATA
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

    # Ensure summary column exists
    if 'summary' not in df.columns:
        df['summary'] = None

    # FIND STARTING INDICES - only check for empty summaries
    empty_mask = (df['summary'].isna() | (df['summary'].astype(str).str.strip() == ""))
    
    indices_to_process = df.index[empty_mask].tolist()
    total_to_do = len(indices_to_process)

    if total_to_do == 0:
        print("All items have been processed!")
        return

    print(f"Found {total_to_do} empty items. First target index: {indices_to_process[0]}")

    # RUN SCRAPING & SUMMARIZING
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
            viewport={'width': 1920, 'height': 1080}
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TABS)

        async def sem_task(url, current_count, total):
            async with semaphore:
                print(f"[{current_count}/{total}] Scraping: {url}")
                return await get_content(context, url)

        for i in range(0, total_to_do, BATCH_SIZE):
            batch_indices = indices_to_process[i : i + BATCH_SIZE]
            batch_urls = df.loc[batch_indices, 'Insured Website'].tolist()
            
            print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ({len(batch_indices)} items) ---")
            
            # Scrape Batch into local variable only
            scraped_results = await asyncio.gather(
                *[sem_task(u, i + idx + 1, total_to_do) for idx, u in enumerate(batch_urls)]
            )
            
            # Summarize Batch and update DF
            print(f"Summarizing batch...")
            for idx, text in enumerate(scraped_results):
                real_idx = batch_indices[idx]
                summary = await async_summarize(text)
                df.loc[real_idx, 'summary'] = summary
            
            # Overwrite the CSV
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Checkpoint saved: {i + len(batch_indices)}/{total_to_do} items completed.")

        await browser.close()

    print(f"\nTask Complete! Final data saved to {OUTPUT_FILE}")
