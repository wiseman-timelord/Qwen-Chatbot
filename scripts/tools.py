# scripts/tools.py
# Qwen-Chatbot: Windows 10 / Python 3.12 / Gradio 5.x
"""
Centralized tools module for web search and TTS.

Search Tools:
- Web Search: Comprehensive multi-source web search with parallel page fetching

TTS Tools:
- Text-to-Speech using Kokoro TTS (kokoro>=0.9.4) on all supported platforms
- G2P via misaki[en] — no espeak dependency on any platform
- Audio playback via winsound
"""

import os
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime
import asyncio
import re
from urllib.parse import urlparse, urljoin, quote_plus
from typing import List, Dict, Tuple, Optional, Any
import time
import tempfile

# Lazy import to avoid circular dependency
import scripts.configure as cfg

# =============================================================================
# WEB SEARCH - COMPREHENSIVE MULTI-SOURCE SEARCH
# =============================================================================

class WebSearchEngine:
    """
    Hybrid multi-source web search optimised for current-events / research queries.

    Design goals:
    - Prefer real news over calendar/observance pages.
    - Bias hard toward recency when the user asks about "previous N days",
      "latest", a specific date, or a conflict/war.
    - Issue complementary query variants and merge, so a single awkward
      extraction does not collapse the whole search.
    - Soft-fail on anti-bot walls; fall back to snippets rather than aborting.
    """

    # Domain quality scores for ranking (news orgs elevated)
    DOMAIN_QUALITY = {
        'high': [
            'reuters.com', 'bbc.com', 'bbc.co.uk', 'apnews.com', 'afp.com',
            'nytimes.com', 'theguardian.com', 'washingtonpost.com', 'wsj.com',
            'ft.com', 'bloomberg.com', 'aljazeera.com', 'dw.com', 'npr.org',
            'economist.com', 'theatlantic.com', 'axios.com', 'politico.com',
            'gov', 'edu', 'arxiv.org', 'nature.com', 'science.org',
            'understandingwar.org', 'crisisgroup.org', 'jinsa.org',
        ],
        'medium': [
            'cnn.com', 'forbes.com', 'wired.com', 'techcrunch.com',
            'arstechnica.com', 'medium.com', 'substack.com', 'scmp.com',
            'haaretz.com', 'jpost.com', 'timesofisrael.com', 'dawn.com',
            'telegraph.co.uk', 'thetimes.com', 'cbc.ca',
        ],
        'low': [
            'reddit.com', 'quora.com', 'twitter.com', 'x.com', 'facebook.com',
            'pinterest.com', 'tiktok.com', 'instagram.com',
        ],
    }

    # Domains / title patterns that are almost never useful for news research
    NOISE_PATTERNS = [
        r'calendar', r'observance', r'holidays?\s+in', r'national\s+day',
        r'world\s+\w+\s+day', r'teachers?\s+day', r'ozone\s+day',
        r'what\s+day\s+is', r'day\s+of\s+the\s+year', r'today\s+in\s+history',
        r'on\s+this\s+day', r'birthday', r'zodiac',
    ]

    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0',
    ]

    TEMPORAL_KEYWORDS = {
        'latest', 'recent', 'current', 'today', 'yesterday', 'breaking',
        'previous', 'last', 'past', 'week', 'days', 'month', 'updates',
        'developments', 'events', 'timeline', 'since', '2024', '2025',
        '2026', '2027', 'september', 'august', 'july', 'june', 'may',
        'april', 'march', 'february', 'january', 'october', 'november',
        'december',
    }

    CONFLICT_KEYWORDS = {
        'war', 'conflict', 'strike', 'missile', 'attack', 'escalation',
        'ceasefire', 'hostilities', 'invasion', 'airstrike', 'bombardment',
        'iran', 'israel', 'lebanon', 'hezbollah', 'houthi', 'gaza',
    }

    def __init__(self):
        self._user_agent_idx = 0

    def _get_user_agent(self) -> str:
        ua = self.USER_AGENTS[self._user_agent_idx % len(self.USER_AGENTS)]
        self._user_agent_idx += 1
        return ua

    def _get_domain_score(self, url: str) -> int:
        try:
            domain = urlparse(url).netloc.lower().replace('www.', '')
            for d in self.DOMAIN_QUALITY['high']:
                if d in domain:
                    return 4
            for d in self.DOMAIN_QUALITY['medium']:
                if d in domain:
                    return 2
            for d in self.DOMAIN_QUALITY['low']:
                if d in domain:
                    return 0
            return 1
        except Exception:
            return 1

    def _is_noise_result(self, result: Dict) -> bool:
        blob = (result.get('title', '') + ' ' + result.get('snippet', '') + ' ' + result.get('url', '')).lower()
        return any(re.search(p, blob) for p in self.NOISE_PATTERNS)

    def _looks_temporal(self, query: str) -> bool:
        words = set(re.findall(r'[a-z0-9]+', query.lower()))
        return bool(words & self.TEMPORAL_KEYWORDS) or bool(re.search(r'\b20\d{2}\b', query))

    def _looks_conflict(self, query: str) -> bool:
        words = set(re.findall(r'[a-z0-9]+', query.lower()))
        return bool(words & self.CONFLICT_KEYWORDS)

    def _score_search_result(self, result: Dict, query_words: set, prefer_news: bool = False) -> int:
        """Relevance-first ranking. Date is a mild secondary signal only when the
        user query itself is temporal (prefer_news=True). Historical research
        must not be demoted for lacking a 2025/2026 stamp."""
        score = 0
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        url = result.get('url', '').lower()
        date_str = (result.get('date') or '').lower()

        if self._is_noise_result(result):
            return -50

        # Primary: topical relevance
        for word in query_words:
            if len(word) <= 2:
                continue
            if word in title:
                score += 6
            if word in snippet:
                score += 2
            if word in url:
                score += 1

        score += self._get_domain_score(url) * 3

        if len(snippet) > 120:
            score += 2

        # Secondary: only when the user asked for recent/current material
        if prefer_news:
            if result.get('source', '').startswith('ddgs_news'):
                score += 6
            if date_str:
                score += 3
            # Prefer pages that carry any explicit date signal (not a specific year)
            if re.search(r'/20\d{2}[/-]', url) or re.search(r'\b20\d{2}\b', title):
                score += 2

        return score

    def _build_query_variants(self, query: str) -> List[str]:
        """Produce 1–3 complementary queries. Temporal/news variants are only
        added when the user query itself signals recency; historical topics are
        left alone so ranking stays relevance-first."""
        q = query.strip()
        variants = [q]
        lower = q.lower()
        temporal = self._looks_temporal(q)

        # Light cleanup of leftover instructional words (generic)
        core = re.sub(
            r'\b(please|produce|research|find\s+out|most\s+notable|and\s+then|'
            r'with\s+a\s+table|concise\s+notes|write\s+a\s+report)\b',
            ' ', lower, flags=re.I
        )
        core = re.sub(r'\s+', ' ', core).strip()

        if temporal and core and core != lower:
            # User asked for recent/current material — add a news-oriented pass
            news_q = f"{core} news"
            if news_q not in variants and len(news_q) > 8:
                variants.append(news_q)
            if any(w in lower for w in ('latest', 'recent', 'current', 'today', 'breaking')):
                latest_q = f"{core} latest"
                if latest_q not in variants and len(latest_q) > 8:
                    variants.append(latest_q)

        # Deduplicate while preserving order
        seen = set()
        out = []
        for v in variants:
            key = v.lower()
            if key not in seen and len(v) > 5:
                seen.add(key)
                out.append(v)
        return out[:3]

    def _search_duckduckgo_html(self, query: str, max_results: int = 15) -> List[Dict]:
        import requests
        from bs4 import BeautifulSoup

        results = []
        try:
            search_url = "https://html.duckduckgo.com/html/"
            params = {'q': query, 'kl': 'wt-wt'}
            headers = {
                'User-Agent': self._get_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://duckduckgo.com/',
            }
            response = requests.post(search_url, data=params, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')

            for result_div in soup.select('.result')[:max_results]:
                try:
                    title_elem = result_div.select_one('.result__title a')
                    snippet_elem = result_div.select_one('.result__snippet')
                    if not title_elem:
                        continue
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    if 'uddg=' in url:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                        if 'uddg' in parsed:
                            url = parsed['uddg'][0]
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    if title and url and url.startswith('http'):
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'date': '',
                            'source': 'duckduckgo',
                        })
                except Exception:
                    continue
            print(f"[WEB-SEARCH] DDG HTML returned {len(results)} results for '{query[:60]}'")
        except Exception as e:
            print(f"[WEB-SEARCH] DDG HTML error: {e}")
        return results

    def _search_ddgs_api(self, query: str, max_results: int = 10, force_news: bool = False) -> List[Dict]:
        results = []
        try:
            from ddgs import DDGS
            # Fresh instance each call — avoids degraded session state after the first search
            ddgs = DDGS(timeout=18)

            is_news = force_news or self._looks_temporal(query) or self._looks_conflict(query) or any(
                kw in query.lower() for kw in (
                    'news', 'latest', 'current', 'recent', 'today', 'breaking',
                    '2024', '2025', '2026', '2027',
                )
            )

            if is_news:
                try:
                    for r in list(ddgs.news(query, max_results=max_results)):
                        results.append({
                            'title': r.get('title', ''),
                            'url': r.get('url', ''),
                            'snippet': r.get('body', ''),
                            'date': r.get('date', ''),
                            'source': 'ddgs_news',
                        })
                except Exception as e:
                    print(f"[WEB-SEARCH] DDGS news error: {e}")

            try:
                for r in list(ddgs.text(query, max_results=max_results)):
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', ''),
                        'date': '',
                        'source': 'ddgs_text',
                    })
            except Exception as e:
                print(f"[WEB-SEARCH] DDGS text error: {e}")

            print(f"[WEB-SEARCH] DDGS API returned {len(results)} results for '{query[:60]}' (news={is_news})")
        except Exception as e:
            print(f"[WEB-SEARCH] DDGS API error: {e}")
        return results

    def _fetch_page_content(self, url: str, timeout: int = 10) -> Optional[str]:
        # Method 1: newspaper4k
        try:
            from newspaper import Article
            article = Article(url)
            article.download()
            article.parse()
            if article.text and len(article.text) > 100:
                content = article.text[:4500]
                if len(article.text) > 4500:
                    content += "\n[...content truncated...]"
                if article.publish_date:
                    content = f"[Published: {article.publish_date.strftime('%Y-%m-%d')}]\n{content}"
                return content
        except Exception as e:
            err = str(e).lower()
            anti_bot = any(x in err for x in (
                "perimeterx", "cloudflare", "datadome", "akamai", "captcha",
                "access denied", "403", "429", "blocked", "bot", "protected",
                "challenge", "human security",
            ))
            if anti_bot:
                print(f"[WEB-SEARCH] Anti-bot wall on {url} — will use snippet only")
            else:
                print(f"[WEB-SEARCH] newspaper failed for {url}: {e}")

        # Method 2: requests + BS4
        try:
            import requests
            from bs4 import BeautifulSoup
            headers = {
                "User-Agent": self._get_user_agent(),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.google.com/",
                "DNT": "1",
            }
            resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if resp.status_code != 200:
                print(f"[WEB-SEARCH] HTTP {resp.status_code} for {url}")
                return None
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            main = (
                soup.find("article")
                or soup.find("main")
                or soup.find(attrs={"role": "main"})
                or soup.find("div", class_=re.compile(r"(article|content|story|post)", re.I))
            )
            text = (main or soup.body or soup).get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            if text and len(text) > 150:
                content = text[:4500]
                if len(text) > 4500:
                    content += "\n[...content truncated...]"
                return content
        except Exception as e:
            print(f"[WEB-SEARCH] Fallback fetch failed for {url}: {e}")
        return None

    def _fetch_pages_parallel(self, urls: List[str], max_workers: int = 4) -> Dict[str, str]:
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._fetch_page_content, url): url for url in urls}
            try:
                for future in as_completed(future_to_url, timeout=28):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content:
                            results[url] = content
                    except Exception as e:
                        print(f"[WEB-SEARCH] Parallel fetch error for {url}: {e}")
            except FuturesTimeout:
                finished = sum(1 for f in future_to_url if f.done())
                pending = len(future_to_url) - finished
                print(f"[WEB-SEARCH] Fetch timeout: {finished} done, {pending} still running — using partial results")
                for future, url in future_to_url.items():
                    if future.done() and not future.cancelled():
                        try:
                            content = future.result()
                            if content:
                                results[url] = content
                        except Exception:
                            pass
        return results

    def search(self, query: str, max_results: int = 12, deep_fetch: int = 6) -> Dict:
        missing_deps = []
        for pkg, name in (
            ("requests", "requests"),
            ("bs4", "beautifulsoup4"),
            ("newspaper", "newspaper4k"),
            ("lxml", "lxml"),
        ):
            try:
                __import__(pkg if pkg != "bs4" else "bs4")
            except ImportError:
                missing_deps.append(name)
        if missing_deps:
            error_msg = (
                f"Web Search requires missing packages: {', '.join(missing_deps)}. "
                f"Install with: pip install {' '.join(missing_deps)}"
            )
            print(f"[WEB-SEARCH] {error_msg}")
            return {
                'content': f"Web search unavailable: {error_msg}",
                'metadata': {'type': 'web_search', 'query': query, 'error': error_msg, 'sources': []},
            }

        print(f"[WEB-SEARCH] Searching for: {query}")
        variants = self._build_query_variants(query)
        print(f"[WEB-SEARCH] Query variants: {variants}")

        all_results: List[Dict] = []
        prefer_news = self._looks_temporal(query)

        for i, vq in enumerate(variants):
            # Small pause between variants to reduce rate-limit pressure
            if i > 0:
                time.sleep(0.6)
            api_results = self._search_ddgs_api(vq, max_results=max(8, max_results // len(variants) + 2), force_news=prefer_news)
            all_results.extend(api_results)
            if len(all_results) < max_results * 2:
                html_results = self._search_duckduckgo_html(vq, max_results=max(6, max_results // 2))
                all_results.extend(html_results)

        if not all_results:
            return {
                'content': f"No search results found for: {query}\n\nCheck your internet connection or try a different query.",
                'metadata': {'type': 'web_search', 'query': query, 'error': 'No results', 'sources': []},
            }

        # Deduplicate by URL, keep first (usually highest quality source)
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = (r.get('url') or '').strip()
            if not url or url in seen_urls:
                continue
            if self._is_noise_result(r):
                continue
            seen_urls.add(url)
            unique_results.append(r)

        query_words = set(re.findall(r'[a-z0-9]{3,}', query.lower()))
        scored = [(self._score_search_result(r, query_words, prefer_news=prefer_news), r) for r in unique_results]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Drop heavily negative scores (noise)
        scored = [s for s in scored if s[0] > -10]
        top_results = [r for _, r in scored[:deep_fetch]]
        remaining_results = [r for _, r in scored[deep_fetch:max_results]]

        urls_to_fetch = [r['url'] for r in top_results if r.get('url')]
        fetched_content = self._fetch_pages_parallel(urls_to_fetch)

        content_parts = []
        sources = []
        content_parts.append("═══════════════════════════════════════════════════════")
        content_parts.append(f"WEB SEARCH RESULTS: {query}")
        content_parts.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        content_parts.append("═══════════════════════════════════════════════════════\n")

        for i, result in enumerate(top_results, 1):
            url = result.get('url', '')
            title = result.get('title', 'Untitled')
            sources.append({
                'title': title,
                'url': url,
                'fetched': url in fetched_content,
                'type': 'deep',
                'date': result.get('date', ''),
            })
            content_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            content_parts.append(f"📰 ARTICLE {i}: {title}")
            content_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            content_parts.append(f"URL: {url}")
            if result.get('date'):
                content_parts.append(f"Date: {result['date']}")
            if url in fetched_content:
                content_parts.append(f"\n{fetched_content[url]}\n")
            else:
                content_parts.append(f"\n{result.get('snippet', 'No content available')}\n")

        if remaining_results:
            content_parts.append("\n───────────────────────────────────────────────────────")
            content_parts.append("📋 ADDITIONAL SOURCES")
            content_parts.append("───────────────────────────────────────────────────────")
            for i, result in enumerate(remaining_results, 1):
                title = result.get('title', 'Untitled')
                snippet = result.get('snippet', '')
                url = result.get('url', '')
                sources.append({
                    'title': title,
                    'url': url,
                    'fetched': False,
                    'type': 'snippet',
                    'date': result.get('date', ''),
                })
                content_parts.append(f"\n[{i}] {title}")
                content_parts.append(f"    {snippet[:220]}..." if len(snippet) > 220 else f"    {snippet}")
                content_parts.append(f"    <{url}>")

        final_content = "\n".join(content_parts)
        fetched_count = sum(1 for s in sources if s.get('fetched'))
        print(f"[WEB-SEARCH] Complete: {fetched_count} deep fetched, {len(remaining_results)} snippets, {len(sources)} total sources")

        return {
            'content': final_content,
            'metadata': {
                'type': 'web_search',
                'query': query,
                'variants': variants,
                'total_results': len(sources),
                'deep_fetched': fetched_count,
                'sources': sources,
                'error': None,
            },
        }


# Global web search engine instance (re-created lightly; methods are mostly pure)
_web_search_engine = None

def get_web_search_engine() -> WebSearchEngine:
    """Get or create web search engine instance."""
    global _web_search_engine
    if _web_search_engine is None:
        _web_search_engine = WebSearchEngine()
    return _web_search_engine


def web_search(query: str, max_results: int = 12, deep_fetch: int = 6) -> Dict:
    """
    Perform comprehensive hybrid web search.

    Args:
        query:       Search query string
        max_results: Maximum results to consider
        deep_fetch:  Number of pages to fetch full content from

    Returns:
        Dict with 'content' (str) and 'metadata' (dict)
    """
    engine = get_web_search_engine()
    return engine.search(query, max_results, deep_fetch)


def format_web_search_status_for_chat(search_metadata: dict) -> str:
    """Format web search metadata into a readable status string for the chat display."""
    if not search_metadata:
        return ""

    lines = []
    query = search_metadata.get('query', '')
    sources = search_metadata.get('sources', [])
    error = search_metadata.get('error')

    display_query = query[:80] + "..." if len(query) > 80 else query

    if error:
        lines.append(f"🌐 Web Search: \"{display_query}\" — ⚠️ {error}")
    else:
        deep_sources = [s for s in sources if s.get('type') == 'deep']
        snippet_sources = [s for s in sources if s.get('type') == 'snippet']
        fetched = sum(1 for s in deep_sources if s.get('fetched'))

        lines.append(f"🌐 Web Search: \"{display_query}\"")
        lines.append(f"   📰 {fetched}/{len(deep_sources)} articles fetched")
        if snippet_sources:
            lines.append(f"   📋 {len(snippet_sources)} additional snippets")

        for source in deep_sources:
            if source.get('fetched'):
                url = source.get('url', '')
                try:
                    domain = urlparse(url).netloc.replace('www.', '')
                    lines.append(f"      ✓ {domain}")
                except Exception:
                    pass

    return "\n".join(lines)


def format_search_status_for_chat(search_metadata: dict) -> str:
    """Format search metadata for chat display. Delegates to the web search formatter."""
    return format_web_search_status_for_chat(search_metadata)


# =============================================================================
# TTS (TEXT-TO-SPEECH) FUNCTIONS
# =============================================================================

# ---------------------------------------------------------------------------
# Thread management
# ---------------------------------------------------------------------------
_tts_lock = threading.Lock()
_tts_thread = None
_tts_stop_flag = threading.Event()

# ---------------------------------------------------------------------------
# Kokoro pipeline cache
# One KPipeline instance per lang_code kept alive between calls to avoid
# the model-load overhead on every utterance.
# ---------------------------------------------------------------------------
_kokoro_pipelines: dict = {}        # lang_code -> KPipeline
_kokoro_model = None                # single shared KModel instance (weights loaded once)
_kokoro_pipeline_lock = threading.Lock()

# ---------------------------------------------------------------------------
# VOICE CATALOGUE
# ---------------------------------------------------------------------------
# 10 curated voices across American and British English.
# id        — passed verbatim to KPipeline()(text, voice=...)
# lang_code — passed to KPipeline(lang_code=...)  'a'=American  'b'=British
# ---------------------------------------------------------------------------
KOKORO_VOICES = [
    {"id": "af_heart",   "name": "Heart — American Female",   "lang_code": "a", "gender": "female"},
    {"id": "af_bella",   "name": "Bella — American Female",   "lang_code": "a", "gender": "female"},
    {"id": "af_nova",    "name": "Nova — American Female",    "lang_code": "a", "gender": "female"},
    {"id": "af_sky",     "name": "Sky — American Female",     "lang_code": "a", "gender": "female"},
    {"id": "am_adam",    "name": "Adam — American Male",      "lang_code": "a", "gender": "male"},
    {"id": "am_michael", "name": "Michael — American Male",   "lang_code": "a", "gender": "male"},
    {"id": "bf_emma",    "name": "Emma — British Female",     "lang_code": "b", "gender": "female"},
    {"id": "bf_alice",   "name": "Alice — British Female",    "lang_code": "b", "gender": "female"},
    {"id": "bm_george",  "name": "George — British Male",     "lang_code": "b", "gender": "male"},
    {"id": "bm_lewis",   "name": "Lewis — British Male",      "lang_code": "b", "gender": "male"},
]

_VOICE_BY_ID   = {v["id"]:   v for v in KOKORO_VOICES}
_VOICE_BY_NAME = {v["name"]: v for v in KOKORO_VOICES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def get_enabled_voices() -> List[Dict]:
    """Return list of voice dicts filtered by the current TTS pack."""
    if cfg.TTS_ENABLED_VOICES:
        return [v for v in KOKORO_VOICES if v["id"] in cfg.TTS_ENABLED_VOICES]
    # Fallback: all voices (used during initialisation or if config not loaded)
    return KOKORO_VOICES

def _kokoro_cache_dir() -> Path:
    """Local directory where Kokoro stores its downloaded model files."""
    d = Path(__file__).parent.parent / "data" / "tts_models" / "kokoro"
    d.mkdir(parents=True, exist_ok=True)
    return d


def detect_tts_engine() -> str:
    """Return 'kokoro' if the kokoro package is importable, else 'none'."""
    try:
        import kokoro  # noqa: F401
        return "kokoro"
    except ImportError as e:
        print(f"[TTS] Kokoro import failed: {e}")
        print("[TTS] Re-run the installer to repair the Kokoro installation.")
        return "none"


def detect_audio_backend() -> str:
    """Detect audio playback backend. Returns "windows"."""
    return "windows"


def _get_kokoro_voices() -> List[Dict[str, str]]:
    """Return enabled voices, with the configured default voice sorted first."""
    saved_id = getattr(cfg, "TTS_VOICE", None) or getattr(cfg, "TTS_DEFAULT_VOICE_ID", None)
    voices = list(get_enabled_voices()) or list(KOKORO_VOICES)
    if saved_id and saved_id in _VOICE_BY_ID:
        voices = sorted(voices, key=lambda v: (v["id"] != saved_id))
    return voices


def get_voice_choices() -> List[str]:
    """Return voice display names for the UI dropdown (filtered by pack)."""
    voices = get_enabled_voices()
    # Sort so that default voice appears first (already handled by _get_kokoro_voices)
    # but we must also apply the same sorting to enabled voices.
    saved_id = getattr(cfg, "TTS_VOICE", None)
    if saved_id and saved_id in [v["id"] for v in voices]:
        voices = sorted(voices, key=lambda v: (v["id"] != saved_id))
    return [v["name"] for v in voices] if voices else ["No voices available"]


def get_voice_id_by_name(voice_name: str) -> Optional[str]:
    """Resolve a display name to a Kokoro voice ID."""
    entry = _VOICE_BY_NAME.get(voice_name)
    return entry["id"] if entry else None


def verify_tts_voice(voice_id: str) -> tuple[bool, str]:
    """Check that voice_id has a local .pt file in the snapshot.

    Returns (ok: bool, message: str).  Called after the user saves a voice
    selection so the UI can report a missing file immediately.
    """
    if not voice_id:
        return False, "No voice selected."
    snapshot = _find_kokoro_snapshot()
    if snapshot is None:
        return False, "Kokoro model snapshot not found — re-run the installer."
    voice_pt = snapshot / "voices" / f"{voice_id}.pt"
    if voice_pt.is_file():
        return True, f"Voice ready: {voice_id}"
    return False, (
        f"Voice file not installed: {voice_id}.pt\n"
        f"Re-run the installer and select this voice pack, "
        f"or manually run: pipeline.load_single_voice('{voice_id}')"
    )


def get_sample_rate_choices() -> List[int]:
    """Kokoro synthesises at 24 000 Hz; expose common playback rates for the UI."""
    return [24000, 44100, 48000]


# ---------------------------------------------------------------------------
# Pipeline management
# ---------------------------------------------------------------------------

def _find_kokoro_snapshot() -> Optional[Path]:
    """Return the local HuggingFace snapshot directory for hexgrad/Kokoro-82M.

    Looks inside data/tts_models/kokoro/hub/models--hexgrad--Kokoro-82M/snapshots/
    and returns the first (and normally only) snapshot subdirectory found.
    Returns None if not found.
    """
    hub_dir = _kokoro_cache_dir() / "hub" / "models--hexgrad--Kokoro-82M" / "snapshots"
    if not hub_dir.is_dir():
        return None
    snapshots = [d for d in hub_dir.iterdir() if d.is_dir()]
    return snapshots[0] if snapshots else None


def _get_or_create_pipeline(lang_code: str):
    """Return a cached KPipeline for *lang_code*, creating it on first call.

    Both 'a' (American) and 'b' (British) pipelines share the same underlying
    KModel instance; only the G2P dialect differs.  Loading from local snapshot
    paths bypasses huggingface_hub entirely — no network access needed and the
    HF_HUB_OFFLINE flag set by launcher.py is irrelevant.
    """
    global _kokoro_model
    from kokoro import KModel, KPipeline

    with _kokoro_pipeline_lock:
        if lang_code not in _kokoro_pipelines:
            snapshot = _find_kokoro_snapshot()
            if snapshot is None:
                raise FileNotFoundError(
                    "Kokoro model snapshot not found in data/tts_models/kokoro/hub. "
                    "Re-run the installer to download the model."
                )

            config_path = str(snapshot / "config.json")
            model_path  = str(snapshot / "kokoro-v1_0.pth")

            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"Kokoro config.json not found: {config_path}")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Kokoro model weights not found: {model_path}")

            print(f"[TTS] Loading Kokoro pipeline (lang_code='{lang_code}')...")
            print(f"[TTS] Snapshot: {snapshot}")

            # Load model weights once; reuse the same KModel instance across
            # all language pipelines to save ~300 MB of duplicate RAM.
            if _kokoro_model is None:
                _kokoro_model = KModel(
                    repo_id="hexgrad/Kokoro-82M",
                    config=config_path,
                    model=model_path,
                )
                print("[TTS] KModel weights loaded from local snapshot")

            _kokoro_pipelines[lang_code] = KPipeline(
                lang_code=lang_code,
                repo_id="hexgrad/Kokoro-82M",
                model=_kokoro_model,
            )
            print(f"[TTS] Kokoro pipeline ready (lang_code='{lang_code}')")
        return _kokoro_pipelines[lang_code]


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _synthesize_kokoro_to_file(text: str, voice_id: Optional[str] = None) -> Optional[str]:
    """Synthesise *text* with Kokoro and write a WAV to the temp directory.

    Returns the WAV file path on success, None on failure.
    """
    import soundfile as sf
    import numpy as np

    # Resolve voice — user's live TTS_VOICE selection takes priority, then
    # the pack default, then the first enabled voice.  Never fall back to a
    # hardcoded id that may not be installed.
    enabled    = get_enabled_voices()
    enabled_ids = [v["id"] for v in enabled]

    def _pick(vid):
        return _VOICE_BY_ID.get(vid) if vid and vid in _VOICE_BY_ID else None

    entry = (
        _pick(voice_id)
        or _pick(getattr(cfg, "TTS_VOICE", None))
        or _pick(getattr(cfg, "TTS_DEFAULT_VOICE_ID", None))
        or (enabled[0] if enabled else None)
        or KOKORO_VOICES[0]
    )
    effective_id   = entry["id"]
    effective_lang = entry["lang_code"]

    # Resolve the local .pt path — KPipeline skips hf_hub_download when
    # the voice argument ends with '.pt'.
    snapshot = _find_kokoro_snapshot()
    if snapshot is None:
        print("[TTS] Kokoro snapshot not found — re-run installer")
        return None
    voice_pt = snapshot / "voices" / f"{effective_id}.pt"
    if not voice_pt.is_file():
        print(f"[TTS] Voice file not found: {voice_pt}")
        # Fall back to first installed voice rather than silently using wrong voice
        for fallback_id in enabled_ids:
            fb_pt = snapshot / "voices" / f"{fallback_id}.pt"
            if fb_pt.is_file():
                print(f"[TTS] Falling back to installed voice: {fallback_id}")
                entry          = _VOICE_BY_ID[fallback_id]
                effective_id   = fallback_id
                effective_lang = entry["lang_code"]
                voice_pt       = fb_pt
                break
        else:
            print("[TTS] No installed voice .pt files found in snapshot")
            return None

    print(f"[TTS] Kokoro synthesizing: voice={effective_id}, lang={effective_lang}")

    try:
        pipeline = _get_or_create_pipeline(effective_lang)
    except Exception as e:
        print(f"[TTS] Failed to load Kokoro pipeline: {e}")
        return None

    temp_dir = Path(cfg.TEMP_DIR) if cfg.TEMP_DIR else Path(__file__).parent.parent / "data" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = str(temp_dir / f"kokoro_{int(time.time() * 1000)}.wav")

    try:
        chunks = []
        # Pass the full local .pt path — KPipeline detects the .pt suffix and
        # loads from disk, bypassing hf_hub_download entirely.
        for _gs, _ps, audio in pipeline(text, voice=str(voice_pt), speed=1.0):
            if _tts_stop_flag.is_set():
                print("[TTS] Synthesis cancelled")
                return None
            if audio is not None and len(audio) > 0:
                chunks.append(audio)

        if not chunks:
            print("[TTS] Kokoro produced no audio chunks")
            return None

        combined = np.concatenate(chunks)
        sf.write(wav_path, combined, 24000)

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            print("[TTS] Kokoro wrote an empty file")
            return None

        print(f"[TTS] Synthesized -> {wav_path}")
        return wav_path

    except Exception as e:
        print(f"[TTS] Kokoro synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return None


def synthesize_text_to_file(text: str, voice_id: Optional[str] = None) -> Optional[str]:
    """Synthesise *text* to a WAV file without playing it.

    Returns the WAV path or None on failure.
    """
    if cfg.TTS_ENGINE == "none":
        print("[TTS] No TTS engine available")
        return None

    if not text or not text.strip():
        print("[TTS] Empty text, skipping")
        return None

    text = _clean_text_for_tts(text)
    if not text:
        print("[TTS] Text empty after cleaning")
        return None

    max_len = getattr(cfg, "MAX_TTS_LENGTH", 4500)
    if len(text) > max_len:
        print(f"[TTS] Text truncated from {len(text)} to {max_len} chars")
        text = text[:max_len]

    return _synthesize_kokoro_to_file(text, voice_id)


def synthesize_last_response(session_messages: list) -> Optional[str]:
    """Synthesise TTS audio from the last AI response (non-blocking path).

    Returns the WAV path or None on failure.  Does NOT play — call
    play_tts_audio() separately.
    """
    if not getattr(cfg, "TTS_ENABLED", False):
        return None
    if not session_messages:
        return None

    last_response = None
    for msg in reversed(session_messages):
        if msg.get("role") == "assistant":
            last_response = msg.get("content", "")
            break
    if not last_response:
        return None

    text = _clean_text_for_tts(last_response)
    if not text:
        return None

    max_len = getattr(cfg, "MAX_TTS_LENGTH", 4500)
    if len(text) > max_len:
        text = text[:max_len] + "... Response truncated for speech."

    return synthesize_text_to_file(text, getattr(cfg, "TTS_VOICE", None))


# ---------------------------------------------------------------------------
# Audio playback
# ---------------------------------------------------------------------------

def _wav_duration_seconds(file_path: str) -> float:
    """Return duration of a WAV file in seconds (fallback estimate on failure)."""
    try:
        import soundfile as sf
        info = sf.info(file_path)
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        # Kokoro default: 24 kHz mono 16-bit PCM
        size = Path(file_path).stat().st_size
        return max(0.5, (size - 44) / (24000 * 2))
    except Exception:
        return 30.0


def _play_audio_file(file_path: str, output_device: Optional[str] = None):
    """Play audio with interruptible winsound (SND_ASYNC + poll).

    Polls _tts_stop_flag every 100 ms so pause / Emergency Stop can cut
    playback via SND_PURGE. Fully synchronous PlaySound cannot be stopped
    reliably from another thread on all Windows builds.
    """
    try:
        import winsound
        duration = _wav_duration_seconds(file_path)
        winsound.PlaySound(
            file_path,
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
        )
        elapsed = 0.0
        step = 0.1
        while elapsed < duration + 0.15:
            if _tts_stop_flag.is_set():
                winsound.PlaySound(None, winsound.SND_PURGE)
                print("[TTS] Playback purged (stop flag)")
                return
            time.sleep(step)
            elapsed += step
        return
    except Exception as e:
        print(f"[TTS] winsound play error: {e}")
    try:
        from playsound import playsound
        playsound(file_path)
        return
    except Exception:
        pass
    print("[TTS] No Windows audio playback available")


def play_tts_audio(wav_path: str, output_device: Optional[str] = None):
    """Play a synthesised TTS WAV then delete it.

    Blocks the caller until playback ends or stop_speaking() is called.
    Always invoke from a background worker so the Gradio UI stays responsive.
    """
    if not wav_path or wav_path == "__played__":
        return
    try:
        if not Path(wav_path).exists():
            print(f"[TTS] Audio file not found: {wav_path}")
            return
        if _tts_stop_flag.is_set():
            print("[TTS] Playback skipped — stop flag set")
            return
        if output_device is None:
            output_device = getattr(cfg, "SOUND_OUTPUT_DEVICE", "Default Sound Device")
        if output_device == "Default Sound Device":
            output_device = "default"
        print(f"[TTS] Playing audio: {wav_path}")
        _play_audio_file(wav_path, output_device)
        if _tts_stop_flag.is_set():
            print("[TTS] Playback interrupted")
        else:
            print("[TTS] Playback complete")
    except Exception as e:
        print(f"[TTS] Playback error: {e}")
    finally:
        try:
            if Path(wav_path).exists():
                Path(wav_path).unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Threading helpers
# ---------------------------------------------------------------------------

def _speak_thread(text: str, voice_id: Optional[str],
                  output_device: Optional[str], _sample_rate: int):
    """Background thread: synthesise then play."""
    wav = None
    with _tts_lock:
        try:
            wav = _synthesize_kokoro_to_file(text, voice_id)
            if wav and not _tts_stop_flag.is_set():
                _play_audio_file(wav, output_device)
        except Exception as e:
            print(f"[TTS] Speech error: {e}")
        finally:
            if wav and Path(wav).exists():
                try:
                    Path(wav).unlink()
                except Exception:
                    pass


def speak_text(text: str, voice_id: Optional[str] = None,
               output_device: Optional[str] = None,
               sample_rate: Optional[int] = None,
               blocking: bool = False) -> bool:
    """Speak *text* via Kokoro TTS in a background thread."""
    global _tts_thread

    if not getattr(cfg, "TTS_ENABLED", False):
        return False
    if detect_tts_engine() == "none":
        print("[TTS] No TTS engine available")
        return False

    if not voice_id:
        voice_id = getattr(cfg, "TTS_VOICE", None)
    if not output_device:
        output_device = getattr(cfg, "SOUND_OUTPUT_DEVICE", None)
    if not sample_rate:
        sample_rate = getattr(cfg, "SOUND_SAMPLE_RATE", 24000)

    clear_tts_stop()
    _tts_thread = threading.Thread(
        target=_speak_thread,
        args=(text, voice_id, output_device, sample_rate),
        daemon=True,
    )
    _tts_thread.start()
    if blocking:
        _tts_thread.join()
    return True


def stop_speaking():
    """Stop any ongoing TTS synthesis or playback immediately."""
    global _tts_thread
    _tts_stop_flag.set()
    try:
        import winsound
        winsound.PlaySound(None, winsound.SND_PURGE)
        print("[TTS] stop_speaking: SND_PURGE issued")
    except Exception as e:
        print(f"[TTS] stop_speaking: SND_PURGE failed: {e}")
    if _tts_thread and _tts_thread.is_alive():
        _tts_thread.join(timeout=1.5)
    _tts_thread = None


def clear_tts_stop():
    """Clear the TTS stop flag so a new playback can start."""
    _tts_stop_flag.clear()


def is_tts_stop_requested() -> bool:
    """Return True when stop_speaking() has been called and not yet cleared."""
    return _tts_stop_flag.is_set()


def is_speaking() -> bool:
    """Return True if a speak_text background thread is still alive."""
    global _tts_thread
    return _tts_thread is not None and _tts_thread.is_alive()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize_tts() -> bool:
    """Initialise the TTS system.  Called during startup AFTER load_config()."""
    engine  = detect_tts_engine()
    backend = detect_audio_backend()

    cfg.TTS_ENGINE        = engine
    cfg.TTS_AUDIO_BACKEND = backend
    cfg.TTS_ENABLED       = (engine != "none")

    if engine == "kokoro":
        enabled  = get_enabled_voices()
        start_id = getattr(cfg, "TTS_VOICE", None) or getattr(cfg, "TTS_DEFAULT_VOICE_ID", None)
        entry = (_VOICE_BY_ID.get(start_id)
                 or (enabled[0] if enabled else None)
                 or KOKORO_VOICES[0])
        print(f"[TTS] Engine: Kokoro TTS (voice: {entry['name']})")
    else:
        print(f"[TTS] Engine: {engine} (TTS disabled)")
    print(f"[TTS] Audio Backend: {backend}")

    voices = get_enabled_voices()
    if voices:
        # First, try to honour the saved config
        saved_id    = getattr(cfg, "TTS_VOICE",      None)
        saved_name  = getattr(cfg, "TTS_VOICE_NAME", None)
        voice_ids   = [v["id"]   for v in voices]
        voice_names = [v["name"] for v in voices]

        if saved_id and saved_id in voice_ids:
            entry = _VOICE_BY_ID[saved_id]
            cfg.TTS_VOICE      = entry["id"]
            cfg.TTS_VOICE_NAME = entry["name"]
            print(f"[TTS] Voice from config: {cfg.TTS_VOICE_NAME} ({cfg.TTS_VOICE})")
        elif saved_name and saved_name in voice_names:
            entry = _VOICE_BY_NAME[saved_name]
            cfg.TTS_VOICE      = entry["id"]
            cfg.TTS_VOICE_NAME = entry["name"]
            print(f"[TTS] Voice from config: {cfg.TTS_VOICE_NAME} ({cfg.TTS_VOICE})")
        else:
            # Use the pack's default voice if available, otherwise first enabled voice
            default_id = getattr(cfg, "TTS_DEFAULT_VOICE_ID", None)
            if default_id and default_id in voice_ids:
                entry = _VOICE_BY_ID[default_id]
            else:
                entry = voices[0]
            cfg.TTS_VOICE      = entry["id"]
            cfg.TTS_VOICE_NAME = entry["name"]
            print(f"[TTS] Default voice from pack: {cfg.TTS_VOICE_NAME} ({cfg.TTS_VOICE})")
    else:
        cfg.TTS_VOICE      = None
        cfg.TTS_VOICE_NAME = "No voices available"
        print("[TTS] No voices enabled")

    return engine != "none"


def get_tts_status() -> str:
    """Return a human-readable TTS status string for the UI."""
    enabled = getattr(cfg, "TTS_ENABLED", False)
    if enabled:
        voice = getattr(cfg, "TTS_VOICE_NAME", "Default")
        return f"TTS: ON (Kokoro — {voice})"
    return "TTS: OFF (Kokoro)"


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_text_for_tts(text: str) -> str:
    """Shared text-cleaning pipeline applied before any TTS synthesis."""
    text = re.sub(r"^AI-Chat:\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?m)^Thinking[.\s]+\r?\n?", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"!\[.*?\]\([^)]+\)", "", text)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\*",   "", text)
    text = re.sub(r"~~",   "", text)
    text = re.sub(r"(?<!\w)_|_(?!\w)", "", text)
    text = re.sub(r"[#•→⇒★☆]|[-=]{2,}", " ", text)
    text = re.sub(r"[^\w\s.,!?;:\'\"()-]", " ", text)
    text = text.replace("'", "")   # strip apostrophes so Kokoro reads "don't" as "dont" not "don t"
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public helper used by display.py
# ---------------------------------------------------------------------------

def speak_last_response(session_messages: list) -> str:
    """Speak the last AI response from session messages.

    Returns a status string.
    """
    if not getattr(cfg, "TTS_ENABLED", False):
        return "TTS is disabled"
    if not session_messages:
        return "No messages to speak"

    last_response = None
    for msg in reversed(session_messages):
        if msg.get("role") == "assistant":
            last_response = msg.get("content", "")
            break

    if not last_response:
        return "No AI response to speak"

    text = _clean_text_for_tts(last_response)
    if not text:
        return "Response has no speakable content after cleaning"

    max_len = getattr(cfg, "MAX_TTS_LENGTH", 4500)
    if len(text) > max_len:
        text = text[:max_len] + "... Response truncated for speech."

    voice_id      = getattr(cfg, "TTS_VOICE",           None)
    output_device = getattr(cfg, "SOUND_OUTPUT_DEVICE",  None)
    sample_rate   = getattr(cfg, "SOUND_SAMPLE_RATE",    24000)

    if speak_text(text, voice_id, output_device, sample_rate):
        return "Speaking response..."
    return "Failed to start speech"