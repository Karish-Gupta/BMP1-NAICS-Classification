from playwright.async_api import TimeoutError as PlaywrightTimeout
import asyncio
import re

import urllib.parse
import base64

# Global Constants
MAX_CHARS = 2500         # Max characters to save per site
TIMEOUT = 25000          # 25 seconds
DEFAULT_RESULTS = 5

async def decode_bing_url(url: str) -> str:
    try:
        u = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(u.query)
        param = None
        for key in ('u', 'uddg'):
            if key in qs:
                param = qs[key][0]
                break
        if param:
            if param.startswith('a1'):
                b64 = param[2:]
                try:
                    return base64.b64decode(b64).decode('utf8')
                except Exception:
                    return urllib.parse.unquote(param)
            try:
                return urllib.parse.unquote(param)
            except Exception:
                return param
    except Exception:
        pass
    return url

async def clean_text(text: str) -> str:
    if not text:
        return ''
    # remove scripts and styles
    text = re.sub(r'<script[\s\S]*?>[\s\S]*?<\/script>', ' ', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?>[\s\S]*?<\/style>', ' ', text, flags=re.I)
    # strip tags and entities
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text.lower()

async def top_n_words_from_texts(texts, n: int = 10):
    counts = {}
    for t in texts:
        cleaned = await clean_text(t)
        tokens = re.split(r"\s+", cleaned)
        for tok in tokens:
            tok = tok.strip()
            if not tok or len(tok) < 3:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    sorted_words = sorted(counts.items(), key=lambda x: -x[1])
    return [{'word': w, 'count': c} for w, c in sorted_words[:n]]

async def fetch_page_text(context, url: str) -> str:
    # try request first
    try:
        res = await context.request.get(url, timeout=30000)
        if res.ok:
            ctype = (res.headers.get('content-type') or '').lower()
            text = await res.text()
            if 'html' in ctype or re.search(r'</?html', text, re.I):
                return await clean_text(text)
            return await clean_text(text)
    except Exception:
        pass

    # fallback to real page
    page = await context.new_page()
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
        try:
            doc_text = await page.evaluate("() => document.documentElement && document.documentElement.innerText ? document.documentElement.innerText : ''")
            return await clean_text(doc_text)
        except Exception:
            html = await page.content()
            return await clean_text(html)
    except Exception:
        return ''
    finally:
        await page.close()

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