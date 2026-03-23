from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from webscraping.scraping_utils import fetch_page_text
from webscraping.scraping_utils import get_content
from webscraping.scraping_utils import decode_bing_url
from webscraping.qwen import QwenLLM
import pandas as pd
import asyncio
import logging
import warnings
import urllib.parse
import base64

# Suppress the specific transformers logging bug and deprecation warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# CONFIG
INPUT_FILE = 'data/NAICS_data_with_websites.csv'
OUTPUT_FILE = 'data/website_summaries.csv'
MAX_CONCURRENT_TABS = 5

# Initialize LLM
qwen = QwenLLM()

def summarize_text(text):
    """Sync wrapper for LLM call to handle local processing safely."""
    if not isinstance(text, str) or len(text.strip()) < 50:
        return "Insufficient content for summary."
    
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

    # Combine columns into "search_term"
    df['search_term'] = (
        df['INSURED_NM'].fillna('') + ' ' +
        df['INSURED_ADDR_LN_1'].fillna('') + ' ' +
        df['INSURED_ADDR_LN_2'].fillna('') + ' ' +
        df['INSURED_CTY_NM'].fillna('')
    ).str.strip()

    # Prepare a column for summaries
    df['summary'] = ''

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )

        for idx, row in df.iterrows():
            page = await context.new_page()
            search_url = 'https://www.bing.com/search?q=' + urllib.parse.quote(row['search_term'])
            await page.goto(search_url, wait_until='domcontentloaded')

            anchors = await page.query_selector_all('li.b_algo h2 a')
            results = []
            for a in anchors[:MAX_CONCURRENT_TABS]:  # Limit to top MAX_CONCURRENT_TABS results per search term
                href = await a.get_attribute('href')
                text = await a.inner_text()
                results.append({'href': href, 'text': text})
            decoded = [{'title': r['text'], 'url': decode_bing_url(r['href'])} for r in results]

            page_texts = []
            for a in decoded:
                try:
                    # fetch_page_text may need to be async as well; if not, keep as is
                    txt = await fetch_page_text(context, a['url']) if asyncio.iscoroutinefunction(fetch_page_text) else fetch_page_text(context, a['url'])
                    page_texts.append(txt)
                    print('Fetched:', a['url'])
                except Exception as e:
                    print('Failed to fetch', a['url'], e)

            # Concatenate all fetched texts for summarization
            all_text = '\n'.join(page_texts)
            print("Summarizing with Local LLM (this may take a minute)...")
            summary = summarize_text(all_text)
            df.at[idx, 'summary'] = summary
            await page.close()

        await browser.close()

    # Save Results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Finished! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())