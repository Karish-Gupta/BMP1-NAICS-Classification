import asyncio
import pandas as pd
import logging
import warnings
from playwright.async_api import async_playwright
from webscraping.scraping_utils import get_content
from webscraping.qwen import QwenLLM

# Suppress logging noise
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# CONFIG
INPUT_FILE = 'data/NAICS_data_with_websites.csv'
OUTPUT_FILE = 'website_summaries.csv'
MAX_CONCURRENT_TABS = 5  # How many tabs open at once
BATCH_SIZE = 50          # How many URLs to process before pausing/saving

# Initialize LLM
qwen = QwenLLM()

async def async_summarize(text):
    """Wraps the LLM call to prevent blocking the event loop."""
    if not isinstance(text, str) or len(text.strip()) < 50:
        return "Insufficient content for summary."
    try:
        # Run the sync LLM call in a thread pool to keep the loop moving
        return await asyncio.to_thread(qwen.invoke, text)
    except Exception as e:
        return f"Summary failed: {str(e)}"

async def main():
    print("--- Starting Scraper and Summarizer ---")
    
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Successfully loaded {len(df)} rows from {INPUT_FILE}")
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    urls = df['Insured Website'].tolist()
    all_scraped_content = []
    all_summaries = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Use a single context, but we will manage page closure strictly
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
            viewport={'width': 1920, 'height': 1080}
        )

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TABS)

        async def sem_task(url, index):
            async with semaphore:
                print(f"[{index+1}/{len(urls)}] Scraping: {url}")
                return await get_content(context, url)

        # Process in Batches to prevent memory crashes
        for i in range(0, len(urls), BATCH_SIZE):
            batch_urls = urls[i : i + BATCH_SIZE]
            print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ({len(batch_urls)} URLs) ---")
            
            # Scrape Batch
            batch_results = await asyncio.gather(
                *[sem_task(u, i + idx) for idx, u in enumerate(batch_urls)]
            )
            all_scraped_content.extend(batch_results)

            # Summarize Batch (Sequential to avoid OOM on Local GPU/CPU)
            print(f"Summarizing Batch {i//BATCH_SIZE + 1}...")
            for idx, text in enumerate(batch_results):
                current_url = batch_urls[idx]
                summary = await async_summarize(text)
                all_summaries.append(summary)
            
            # 3. Intermediate Save (Optional but recommended for large datasets)
            temp_df = df.iloc[:len(all_summaries)].copy()
            temp_df['scraped_text'] = all_scraped_content
            temp_df['summary'] = all_summaries
            temp_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Progress saved to {OUTPUT_FILE}")

        await browser.close()

    print(f"\nTask Complete! Final data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())