#!/usr/bin/env node

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const DEFAULT_RESULTS = 5;

function parseArgs() {
  const argv = process.argv.slice(2);
  if (argv.length < 2) {
    console.error('Usage: node scripts/top5.js "Business Name" "Address" [--results=5]');
    process.exit(2);
  }
  const name = argv[0];
  const address = argv[1];
  let results = DEFAULT_RESULTS;
  argv.slice(2).forEach(a => { if (a.startsWith('--results=')) results = Number(a.split('=')[1]) || DEFAULT_RESULTS; });
  return { name, address, results };
}

function decodeBingUrl(url) {
  try {
    const u = new URL(url);
    const param = u.searchParams.get('u') || u.searchParams.get('uddg');
    if (param) {
      if (param.startsWith('a1')) {
        const b64 = param.slice(2);
        try { return Buffer.from(b64, 'base64').toString('utf8'); } catch (_) { return decodeURIComponent(param); }
      }
      try { return decodeURIComponent(param); } catch (_) { return param; }
    }
  } catch (_) {}
  return url;
}

const STOPWORDS = new Set([
  'the','and','for','with','that','this','from','have','will','are','was','were','but','not','you','your',
  'our','they','their','them','what','which','when','where','why','how','than','then','about','into','over',
  'also','can','may','these','those','such','into','more','most','some','other','each','use','used','using',
  'business','company','llc','inc','co','corporation','html','head','meta','div','span','class','id','href','http','https','www','com','org','net',
  'wikipedia','facebook','twitter','linkedin','instagram','youtube','tiktok','pinterest','snapchat','github','stackoverflow',
  'editor','blog','news','press','contact','about','terms','privacy','policy','cookie','cookies','support','help',
  'name','address','phone','email','location','map','directions','hours'
]);

function cleanText(text) {
  if (!text) return '';
  return text
    .replace(/<script[\s\S]*?>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]*>/g, ' ')
    .replace(/&[a-zA-Z0-9#]+;/g, ' ')
    .replace(/[^a-zA-Z\s]/g, ' ')
    .toLowerCase();
}

function topNWordsFromTexts(texts, n = 10) {
  const counts = new Map();
  for (const t of texts) {
    const cleaned = cleanText(t);
    const tokens = cleaned.split(/\s+/).filter(Boolean).map(s => s.trim());
    for (const tok of tokens) {
      if (tok.length < 3) continue;
      if (STOPWORDS.has(tok)) continue;
      counts.set(tok, (counts.get(tok) || 0) + 1);
    }
  }
  return Array.from(counts.entries()).sort((a,b) => b[1]-a[1]).slice(0,n).map(([w,c]) => ({word:w,count:c}));
}

async function fetchPageText(context, url) {
  // prefer request.get to avoid execution contexts; fall back to page navigation if needed
  try {
    const res = await context.request.get(url, { timeout: 30000 });
    if (res && res.ok()) {
      const ctype = (res.headers()['content-type'] || '').toLowerCase();
      const text = await res.text();
      // if content-type suggests HTML, strip tags; else return raw text
      if (ctype.includes('html') || /<\/?html/i.test(text)) return cleanText(text);
      return cleanText(text);
    }
  } catch (e) {
    // continue to page fallback
  }

  // fallback: open a real page and extract innerText
  const p = await context.newPage();
  try {
    await p.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
    try {
      const docText = await p.evaluate(() => document.documentElement && document.documentElement.innerText ? document.documentElement.innerText : '');
      return cleanText(docText);
    } catch (_) {
      const html = await p.content();
      return cleanText(html);
    }
  } catch (e) {
    return '';
  } finally {
    await p.close();
  }
}

async function run() {
  const { name, address, results } = parseArgs();
  const query = `${name} ${address}`;
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36' });
  const page = await context.newPage();

  const searchUrl = 'https://www.bing.com/search?q=' + encodeURIComponent(query);
  await page.goto(searchUrl, { waitUntil: 'domcontentloaded' });

  const anchors = await page.$$eval('li.b_algo h2 a', nodes => nodes.map(n => ({ href: n.href, text: n.innerText })) );
  const topAnchors = anchors.slice(0, results);
  const decoded = topAnchors.map(a => ({ title: a.text, url: decodeBingUrl(a.href) }));

  // fetch each URL and collect cleaned page text
  const pageTexts = [];
  for (const a of decoded) {
    try {
      const txt = await fetchPageText(context, a.url);
      pageTexts.push(txt);
      console.log('Fetched:', a.url);
    } catch (e) {
      console.warn('Failed to fetch', a.url, e.message);
    }
  }

  await browser.close();

  const topWords = topNWordsFromTexts(pageTexts, 10);
  const out = { query, results: decoded, topWords };
  console.log(JSON.stringify(out, null, 2));

  try { fs.writeFileSync(path.resolve(process.cwd(), 'top5-output.json'), JSON.stringify(out, null, 2), 'utf8'); } catch (e) {}
}

run().catch(err => { console.error(err); process.exit(1); });
