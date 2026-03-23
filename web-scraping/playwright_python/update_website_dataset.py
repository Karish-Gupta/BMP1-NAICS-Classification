import re
from collections import Counter

import numpy as np
import pandas as pd
import os
from playwright.sync_api import sync_playwright

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


def add_top_words_column(df: pd.DataFrame, url_column: str = 'Insured Website', words_column: str = 'words') -> pd.DataFrame:
    
    # ensure words column is string/object dtype to accept comma-separated terms
    if words_column not in df.columns:
        df[words_column] = ''
    else:
        df[words_column] = df[words_column].astype('string')

    for idx, row in df.iterrows():
        url = row.get(url_column)
        if pd.isna(url) or not str(url).strip():
            continue
        try:
            df.at[idx, words_column] = ','.join(top_words_from_url(str(url)))
        except Exception as exc:
            print(f"failed to fetch words for {url}: {exc}")
            df.at[idx, words_column] = ''
    return df


if __name__ == "__main__":
    print("Starting Script")
    df = pd.read_csv('data/NAICS_data_with_websites.csv')
    
    # only process a subset for speed when testing
    df = df[:20]

    df = add_top_words_column(df)
    print(df[['Insured Website', 'words']].head())
    df.to_csv('../data/website_dataset.csv', index=False)
    print('Done')
<<<<<<< HEAD

def import_csv():
    df = pd.read_csv('NAICS_data_with_websites.csv')
    return df


if __name__ == "__main__":
    _main_()