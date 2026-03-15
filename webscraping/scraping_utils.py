from playwright.async_api import TimeoutError as PlaywrightTimeout
import asyncio
import re

# Global Constants
MAX_CHARS = 2500         # Max characters to save per site
TIMEOUT = 25000          # 25 seconds

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