from playwright.async_api import async_playwright
from webscraping.scraping_utils import get_content
from webscraping.qwen import QwenLLM
import pandas as pd
import asyncio

# CONFIG
INPUT_FILE = 'data/NAICS_data_with_websites.csv'
OUTPUT_FILE = 'website_summaries.csv'
MAX_CONCURRENT_TABS = 5 

# Initialize LLM
qwen = QwenLLM()

def summarize_text(text):
    """Sync wrapper for LLM call to handle local processing safely."""
    if not text or len(text) < 50:
        return "No sufficient content found to summarize."
    
    try:
        # Assuming QwenLLM has a generate or invoke method
        return qwen.invoke(text)
    except Exception as e:
        return f"Summary failed: {str(e)}"

async def main():
    print("Starting Scraper and Summarizer...")
    
    try:
        # Load data - processing a small batch for testing is recommended
        df = pd.read_csv(INPUT_FILE).head(10)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # SCRAPING (Parallel)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
            viewport={'width': 1920, 'height': 1080}
        )

        urls = df['Insured Website'].tolist()
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TABS)

        async def sem_task(url):
            async with semaphore:
                # get_content is the paragraph scraper we built
                return await get_content(context, url)

        print(f"Scraping {len(urls)} URLs...")
        scraped_results = await asyncio.gather(*[sem_task(u) for u in urls])
        await browser.close()

    df['scraped_text'] = scraped_results

    # SUMMARIZING (Sequential)
    print("Summarizing with Local LLM (this may take a minute)...")
    
    summaries = []
    for i, text in enumerate(df['scraped_text']):
        print(f"[{i+1}/{len(df)}] Summarizing {df['Insured Website'].iloc[i]}...")
        summary = summarize_text(text)
        summaries.append(summary)

    df['summary'] = summaries

    # Save Results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Finished! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())