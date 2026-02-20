#!/usr/bin/env python3

from playwright.sync_api import sync_playwright
import argparse
import base64
import re
import json
import sys
import os
import urllib.parse

DEFAULT_RESULTS = 5

STOPWORDS = {
    'the','and','for','with','that','this','from','have','will','are','was','were','but','not','you','your',
    'our','they','their','them','what','which','when','where','why','how','than','then','about','into','over',
    'also','can','may','these','those','such','into','more','most','some','other','each','use','used','using',
    'business','company','llc','inc','co','corporation','html','head','meta','div','span','class','id','href','http','https','www','com','org','net',
    'wikipedia','facebook','twitter','linkedin','instagram','youtube','tiktok','pinterest','snapchat','github','stackoverflow',
    'editor','blog','news','press','contact','about','terms','privacy','policy','cookie','cookies','support','help',
    'name','address','phone','email','location','map','directions','hours'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape top N Bing search results for a business name + address and extract common words."
    )
    parser.add_argument('name', help='business name')
    parser.add_argument('address', help='address string')
    parser.add_argument('--results', type=int, default=DEFAULT_RESULTS,
                        help=f'number of search results to analyze (default {DEFAULT_RESULTS})')
    return parser.parse_args()


def decode_bing_url(url: str) -> str:
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


def clean_text(text: str) -> str:
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


def top_n_words_from_texts(texts, n: int = 10):
    counts = {}
    for t in texts:
        cleaned = clean_text(t)
        tokens = re.split(r"\s+", cleaned)
        for tok in tokens:
            tok = tok.strip()
            if not tok or len(tok) < 3:
                continue
            if tok in STOPWORDS:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    sorted_words = sorted(counts.items(), key=lambda x: -x[1])
    return [{'word': w, 'count': c} for w, c in sorted_words[:n]]


def fetch_page_text(context, url: str) -> str:
    # try request first
    try:
        res = context.request.get(url, timeout=30000)
        if res.ok:
            ctype = (res.headers.get('content-type') or '').lower()
            text = res.text()
            if 'html' in ctype or re.search(r'</?html', text, re.I):
                return clean_text(text)
            return clean_text(text)
    except Exception:
        pass

    # fallback to real page
    page = context.new_page()
    try:
        page.goto(url, wait_until='domcontentloaded', timeout=30000)
        try:
            doc_text = page.evaluate("() => document.documentElement && document.documentElement.innerText ? document.documentElement.innerText : ''")
            return clean_text(doc_text)
        except Exception:
            html = page.content()
            return clean_text(html)
    except Exception:
        return ''
    finally:
        page.close()


def main():
    args = parse_args()
    query = f"{args.name} {args.address}"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = context.new_page()
        search_url = 'https://www.bing.com/search?q=' + urllib.parse.quote(query)
        page.goto(search_url, wait_until='domcontentloaded')

        anchors = page.query_selector_all('li.b_algo h2 a')
        results = []
        for a in anchors[: args.results]:
            href = a.get_attribute('href')
            text = a.inner_text()
            results.append({'href': href, 'text': text})
        decoded = [{'title': r['text'], 'url': decode_bing_url(r['href'])} for r in results]

        page_texts = []
        for a in decoded:
            try:
                txt = fetch_page_text(context, a['url'])
                page_texts.append(txt)
                print('Fetched:', a['url'])
            except Exception as e:
                print('Failed to fetch', a['url'], e)

        browser.close()

    top_words = top_n_words_from_texts(page_texts, 10)
    out = {'query': query, 'results': decoded, 'topWords': top_words}
    print(json.dumps(out, indent=2))
    try:
        with open(os.path.join(os.getcwd(), 'top5-output.json'), 'w', encoding='utf8') as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass


if __name__ == '__main__':
    main()
