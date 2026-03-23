import asyncio
import pandas as pd
import logging
import warnings
import os
from playwright.async_api import async_playwright
from webscraping.scraping_utils import get_content
from webscraping.qwen import QwenLLM

# Suppress logging noise
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# CONFIG
INPUT_FILE = 'data/split_data/test.csv'
OUTPUT_FILE = 'website_summaries_test.csv'
MAX_CONCURRENT_TABS = 5 
BATCH_SIZE = 20 

# Initialize LLM
qwen = QwenLLM()

async def async_summarize(text):
    if not isinstance(text, str) or len(text.strip()) < 50:
        return "Insufficient content for summary."
    try:
        return await asyncio.to_thread(qwen.invoke, text)
    except Exception as e:
        return f"Summary failed: {str(e)}"

async def main():
    print("--- Starting Scraper and Summarizer ---")
    
    # LOAD DATA
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing output. Loading {OUTPUT_FILE}...")
        df = pd.read_csv(OUTPUT_FILE)
    else:
        if not os.path.exists(INPUT_FILE):
            print(f"Error: {INPUT_FILE} not found.")
            return
        print(f"Starting fresh from {INPUT_FILE}...")
        df = pd.read_csv(INPUT_FILE)

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

if __name__ == "__main__":
    asyncio.run(main())