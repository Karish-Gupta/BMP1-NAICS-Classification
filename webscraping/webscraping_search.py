import asyncio
import urllib.parse
import pandas as pd
from playwright.async_api import async_playwright
from webscraping.qwen import QwenLLM

# CONFIGURATION
INPUT_FILE = 'train_with_summaries.csv'
OUTPUT_FILE = 'business_data_with_summaries.csv'
MAX_TEXT_LENGTH = 2500  # Prevent overloading your LLM's context window

qwen = QwenLLM()

def call_llm_for_summary(scraped_text):
    """
    Call local llm for summarization
    """
    # Simulated response
    if not scraped_text or len(scraped_text) < 50:
        return "Not enough information to summarize."
    
    return qwen.invoke(scraped_text)

async def process_businesses():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    async with async_playwright() as p:
        # Launching headless=True so it runs silently in the background
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )

        for index, row in df.iterrows():
            # Skip if a summary already exists
            if pd.notna(row.get('summary')) and str(row['summary']).strip() != "":
                continue

            business_name = str(row['Business_Name'])
            address = str(row['FULL_ADDRESS'])
            search_query = f"{business_name}, {address}"
            
            print(f"Processing ({index + 1}/{len(df)}): {business_name}")

            page = await context.new_page()
            try:
                # 1. Search using DuckDuckGo HTML (much easier to scrape than Google/Bing)
                encoded_query = urllib.parse.quote(search_query)
                search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
                await page.goto(search_url, wait_until="domcontentloaded", timeout=15000)

                # 2. Find the first organic search result link
                link_locator = page.locator('.result__url').first
                if await link_locator.count() == 0:
                    df.at[index, 'summary'] = "No search results found."
                    continue

                target_url = await link_locator.get_attribute('href')
                
                # DuckDuckGo sometimes wraps URLs in a redirect; this unwraps it
                if target_url and target_url.startswith('//duckduckgo.com/l/?'):
                    parsed_url = urllib.parse.urlparse(target_url)
                    target_url = urllib.parse.parse_qs(parsed_url.query).get('uddg', [None])[0]

                if not target_url:
                    df.at[index, 'summary'] = "Could not extract target URL."
                    continue

                print(f"  -> Scraping: {target_url}")
                
                # 3. Visit the actual business website
                await page.goto(target_url, wait_until='domcontentloaded', timeout=15000)

                # 4. Extract all readable text from the body
                text_content = await page.evaluate('document.body.innerText')
                
                # Clean up whitespace and truncate to fit LLM limits
                cleaned_text = " ".join(text_content.split())[:MAX_TEXT_LENGTH]

                # 5. Send to LLM
                summary = call_llm_for_summary(business_name, cleaned_text)
                df.at[index, 'summary'] = summary
                print("  -> Summary generated.")

            except Exception as e:
                print(f"  -> Error processing {business_name}: {e}")
                df.at[index, 'summary'] = f"Error during scraping: {str(e)}"
            finally:
                await page.close()

        await browser.close()

    # Save the updated DataFrame back to a CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinished! Data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(process_businesses())