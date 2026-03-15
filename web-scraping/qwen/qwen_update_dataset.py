import re
from collections import Counter

import numpy as np
import pandas as pd
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
from huggingface_hub import InferenceClient

STOPWORDS = {
    'the','and','for','with','that','this','from','have','will','are','was','were','but','not','you','your',
    'our','they','their','them','what','which','when','where','why','how','than','then','about','into','over',
    'also','can','may','these','those','such','into','more','most','some','other','each','use','used','using',
    'business','company','llc','inc','co','corporation','html','head','meta','div','span','class','id','href','http','https','www','com','org','net',
    'wikipedia','facebook','twitter','linkedin','instagram','youtube','tiktok','pinterest','snapchat','github','stackoverflow',
    'editor','blog','news','press','contact','about','terms','privacy','policy','cookie','cookies','support','help',
    'name','address','phone','email','location','map','directions','hours'
}

def top_words_from_url(url: str, top_n: int = 10) -> list[str]:
   
    # ensure the URL includes a scheme
    if not re.match(r"^https?://", url):
        url = "http://" + url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=30000)
        text = page.inner_text('body')
        browser.close()

    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if len(t) > 1 and t not in STOPWORDS]
    counts = Counter(tokens)
    return [word for word, _ in counts.most_common(top_n)]


def fetch_page_text(url: str, timeout: int = 30000) -> str:
    if not re.match(r"^https?://", url):
        url = "http://" + url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=timeout)
        text = page.inner_text('body')
        browser.close()

    return text


def describe_url(url: str, client: InferenceClient) -> str:
    page_text = fetch_page_text(url)
    excerpt = page_text[:2000].strip()
    if not excerpt:
        return ""

    prompt = (
        f"Summarize the following business website in one concise sentence:\n\n"
        f"URL: {url}\n"
        f"Text excerpt:\n{excerpt}"
    )
    print("\n--- PROMPT ---\n", prompt)

    response = client.chat.completions.create(
        model="Qwen/Qwen3-4B-Thinking-2507:nscale",
        messages=[
            {"role": "system", "content": "You are a business website summarization assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=80,
        temperature=0.2,
    )

    choice = response.choices[0]
    if hasattr(choice, 'message'):
        content = choice.message.get('content') if isinstance(choice.message, dict) else getattr(choice.message, 'content', '')
    elif isinstance(choice, dict):
        content = choice.get('message', {}).get('content', '') if isinstance(choice.get('message', {}), dict) else ''
        if not content:
            content = choice.get('text', '')
    else:
        content = getattr(choice, 'text', '') if hasattr(choice, 'text') else ''

    answer = str(content).strip()
    print("\n--- ANSWER ---\n", answer)
    return answer


def add_description_column(df: pd.DataFrame, url_column: str = 'Insured Website', description_column: str = 'description', client: InferenceClient = None) -> pd.DataFrame:
    if description_column not in df.columns:
        df[description_column] = ''
    else:
        df[description_column] = df[description_column].astype('string')

    for idx, row in df.iterrows():
        url = row.get(url_column)
        if pd.isna(url) or not str(url).strip():
            continue
        try:
            page_text = fetch_page_text(str(url))
            desc = describe_url(url, client)
            df.at[idx, description_column] = desc
        except Exception as exc:
            print(f"failed to describe {url}: {exc}")
            df.at[idx, description_column] = ''
    return df


def _main_():
    print("Starting Script")
    df = import_csv()
    # only process a subset for speed when testing
    df = df[:20]

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN required in environment for Qwen API.")

    client = InferenceClient(api_key=token)
    df = add_description_column(df, url_column='Insured Website', description_column='description', client=client)

    print(df[['Insured Website', 'description']].head())
    df.to_csv('../data/website_dataset.csv', index=False)
    print('Done')

def import_csv():
    df = pd.read_csv('../data/NAICS_data_with_websites.csv')
    return df


if __name__ == "__main__":
    _main_()