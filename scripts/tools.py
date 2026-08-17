# scripts/tools.py
# v2: Windows 10-11 / Ubuntu 24-25 / Python 3.11-3.12 / Gradio 5.x
"""
Centralized tools module for web search and TTS.

Search Tools:
- Web Search: Comprehensive multi-source web search with parallel page fetching

TTS Tools:
- Text-to-Speech using Kokoro TTS (kokoro>=0.9.4) on all supported platforms
- G2P via misaki[en] — no espeak dependency on any platform
- Audio playback via winsound (Windows) or PipeWire/PulseAudio/ALSA (Linux)
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
    Comprehensive web search engine that:
    1. Searches multiple sources for titles/descriptions
    2. Ranks and selects the most relevant pages
    3. Fetches full content from selected pages in parallel
    4. Extracts and processes content intelligently
    5. Returns a comprehensive summary for the LLM
    """

    # Domain quality scores for ranking
    DOMAIN_QUALITY = {
        'high': ['wikipedia.org', 'britannica.com', 'reuters.com', 'bbc.com', 'bbc.co.uk',
                 'nytimes.com', 'theguardian.com', 'washingtonpost.com', 'apnews.com',
                 'npr.org', 'nature.com', 'science.org', 'gov', 'edu', 'arxiv.org',
                 'sciencedirect.com', 'pubmed.ncbi.nlm.nih.gov', 'smithsonianmag.com'],
        'medium': ['medium.com', 'substack.com', 'cnn.com', 'forbes.com', 'wired.com',
                   'techcrunch.com', 'arstechnica.com', 'theatlantic.com', 'economist.com',
                   'ft.com', 'bloomberg.com', 'wsj.com', 'aljazeera.com', 'dw.com'],
        'low': ['reddit.com', 'quora.com', 'twitter.com', 'facebook.com', 'pinterest.com']
    }

    # User agent rotation for avoiding blocks
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0'
    ]

    def __init__(self):
        self._session = None
        self._user_agent_idx = 0

    def _get_user_agent(self) -> str:
        """Rotate user agents to avoid detection."""
        ua = self.USER_AGENTS[self._user_agent_idx % len(self.USER_AGENTS)]
        self._user_agent_idx += 1
        return ua

    def _get_domain_score(self, url: str) -> int:
        """Score a domain based on quality/trustworthiness."""
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '')
            for high_domain in self.DOMAIN_QUALITY['high']:
                if high_domain in domain:
                    return 3
            for med_domain in self.DOMAIN_QUALITY['medium']:
                if med_domain in domain:
                    return 2
            for low_domain in self.DOMAIN_QUALITY['low']:
                if low_domain in domain:
                    return 0
            return 1
        except:
            return 1

    def _score_search_result(self, result: Dict, query_words: set) -> int:
        """Score a search result based on relevance to query."""
        score = 0
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        url = result.get('url', '')

        for word in query_words:
            if len(word) > 3:
                if word in title:
                    score += 5
                if word in snippet:
                    score += 2

        score += self._get_domain_score(url) * 2

        if result.get('date'):
            score += 3

        if len(snippet) > 150:
            score += 2

        return score

    def _search_duckduckgo_html(self, query: str, max_results: int = 15) -> List[Dict]:
        """
        Search DuckDuckGo via HTML scraping (more comprehensive than API).
        Returns list of {title, url, snippet, date, source}
        """
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
                'Referer': 'https://duckduckgo.com/'
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
                            'source': 'duckduckgo'
                        })
                except Exception:
                    continue

            print(f"[WEB-SEARCH] DDG HTML returned {len(results)} results")

        except Exception as e:
            print(f"[WEB-SEARCH] DDG HTML error: {e}")

        return results

    def _search_ddgs_api(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search using ddgs library as fallback/supplement."""
        results = []
        try:
            from ddgs import DDGS

            ddgs = DDGS(timeout=15)

            news_keywords = ['news', 'latest', 'current', 'recent', 'today', 'breaking',
                             '2024', '2025', '2026', '2027']
            is_news = any(kw in query.lower() for kw in news_keywords)

            if is_news:
                try:
                    ddg_results = list(ddgs.news(query, max_results=max_results))
                    for r in ddg_results:
                        results.append({
                            'title': r.get('title', ''),
                            'url': r.get('url', ''),
                            'snippet': r.get('body', ''),
                            'date': r.get('date', ''),
                            'source': 'ddgs_news'
                        })
                except:
                    pass

            try:
                ddg_results = list(ddgs.text(query, max_results=max_results))
                for r in ddg_results:
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', ''),
                        'date': '',
                        'source': 'ddgs_text'
                    })
            except:
                pass

            print(f"[WEB-SEARCH] DDGS API returned {len(results)} results")

        except Exception as e:
            print(f"[WEB-SEARCH] DDGS API error: {e}")

        return results

    def _fetch_page_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch and extract main content from a web page."""
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()

            if article.text and len(article.text) > 100:
                content = article.text[:4000]
                if len(article.text) > 4000:
                    content += "\n[...content truncated...]"

                if article.publish_date:
                    content = f"[Published: {article.publish_date.strftime('%Y-%m-%d')}]\n{content}"

                return content

        except Exception as e:
            print(f"[WEB-SEARCH] Failed to fetch {url}: {e}")

        return None

    def _fetch_pages_parallel(self, urls: List[str], max_workers: int = 4) -> Dict[str, str]:
        """Fetch multiple pages in parallel. Returns partial results on timeout."""
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._fetch_page_content, url): url for url in urls}

            try:
                for future in as_completed(future_to_url, timeout=25):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content:
                            results[url] = content
                    except Exception as e:
                        print(f"[WEB-SEARCH] Parallel fetch error for {url}: {e}")
            except FuturesTimeout:
                finished = sum(1 for f in future_to_url if f.done())
                pending  = len(future_to_url) - finished
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
        """Perform comprehensive web search with dependency checking."""
        missing_deps = []
        try:
            import requests
        except ImportError:
            missing_deps.append("requests")
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            missing_deps.append("beautifulsoup4")
        try:
            from newspaper import Article
        except ImportError:
            missing_deps.append("newspaper4k")
        try:
            import lxml
        except ImportError:
            missing_deps.append("lxml")

        if missing_deps:
            error_msg = (f"Web Search requires missing packages: {', '.join(missing_deps)}. "
                         f"Install with: pip install {' '.join(missing_deps)}")
            print(f"[WEB-SEARCH] {error_msg}")
            return {
                'content': f"Web search unavailable: {error_msg}",
                'metadata': {'type': 'web_search', 'query': query, 'error': error_msg, 'sources': []}
            }

        # Phase 1: Gather search results — DDGS API (primary) + DDG HTML scrape (supplemental).
        print(f"[WEB-SEARCH] Searching for: {query}")

        all_results = []

        api_results = self._search_ddgs_api(query, max_results)
        all_results.extend(api_results)

        if len(all_results) < max_results:
            html_results = self._search_duckduckgo_html(query, max_results)
            all_results.extend(html_results)

        if not all_results:
            return {
                'content': f"No search results found for: {query}\n\nCheck your internet connection or try a different query.",
                'metadata': {'type': 'web_search', 'query': query, 'error': 'No results', 'sources': []}
            }

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Phase 2: Score and rank results
        query_words = set(query.lower().split())
        scored_results = [(self._score_search_result(r, query_words), r) for r in unique_results]
        scored_results.sort(key=lambda x: x[0], reverse=True)

        top_results = [r for _, r in scored_results[:deep_fetch]]
        remaining_results = [r for _, r in scored_results[deep_fetch:max_results]]

        # Phase 3: Fetch full content from top results
        urls_to_fetch = [r['url'] for r in top_results if r.get('url')]
        fetched_content = self._fetch_pages_parallel(urls_to_fetch)

        # Phase 4: Build final content
        content_parts = []
        sources = []

        content_parts.append(f"═══════════════════════════════════════════════════════")
        content_parts.append(f"WEB SEARCH RESULTS: {query}")
        content_parts.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        content_parts.append(f"═══════════════════════════════════════════════════════\n")

        for i, result in enumerate(top_results, 1):
            url = result.get('url', '')
            title = result.get('title', 'Untitled')

            sources.append({
                'title': title,
                'url': url,
                'fetched': url in fetched_content,
                'type': 'deep'
            })

            content_parts.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            content_parts.append(f"📰 ARTICLE {i}: {title}")
            content_parts.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            content_parts.append(f"URL: {url}")

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
                    'type': 'snippet'
                })

                content_parts.append(f"\n[{i}] {title}")
                content_parts.append(f"    {snippet[:200]}..." if len(snippet) > 200 else f"    {snippet}")
                content_parts.append(f"    <{url}>")

        final_content = "\n".join(content_parts)
        fetched_count = sum(1 for s in sources if s.get('fetched'))

        print(f"[WEB-SEARCH] Complete: {fetched_count} deep fetched, {len(remaining_results)} snippets")

        return {
            'content': final_content,
            'metadata': {
                'type': 'web_search',
                'query': query,
                'total_results': len(sources),
                'deep_fetched': fetched_count,
                'sources': sources,
                'error': None
            }
        }


# Global web search engine instance
_web_search_engine = None

def get_web_search_engine() -> WebSearchEngine:
    """Get or create web search engine instance."""
    global _web_search_engine
    if _web_search_engine is None:
        _web_search_engine = WebSearchEngine()
    return _web_search_engine


def web_search(query: str, max_results: int = 12, deep_fetch: int = 6) -> Dict:
    """
    Perform comprehensive web search.

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
                except:
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
    """Detect audio playback backend.

    Returns: "windows" | "pipewire" | "pulseaudio" | "alsa" | "none"
    """
    if cfg.PLATFORM == "windows":
        return "windows"

    # PipeWire — must be actually running, not just installed
    try:
        result = subprocess.run(["pw-cli", "info", "0"], capture_output=True, timeout=3)
        if result.returncode == 0:
            return "pipewire"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # PulseAudio (or PipeWire-Pulse compatibility layer)
    try:
        result = subprocess.run(["pactl", "info"], capture_output=True, timeout=3)
        if result.returncode == 0:
            if "PipeWire" in result.stdout.decode():
                return "pipewire"
            return "pulseaudio"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # ALSA
    try:
        result = subprocess.run(["aplay", "--version"], capture_output=True, timeout=3)
        if result.returncode == 0:
            return "alsa"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return "none"


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

def _play_audio_file(file_path: str, output_device: Optional[str] = None):
    """Play an audio file using the best available backend."""
    if cfg.PLATFORM == "windows":
        try:
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
            return
        except Exception:
            pass
        try:
            from playsound import playsound
            playsound(file_path)
            return
        except Exception:
            pass
        print("[TTS] No Windows audio playback available")
        return

    # Linux — build environment for user audio session access
    env = os.environ.copy()
    original_uid = os.getuid() if hasattr(os, "getuid") else None
    sudo_user = os.environ.get("SUDO_USER")

    if original_uid == 0 and sudo_user:
        try:
            import pwd
            user_info = pwd.getpwnam(sudo_user)
            user_uid  = user_info.pw_uid
            user_home = user_info.pw_dir
            env["HOME"] = user_home
            env["USER"] = sudo_user
            runtime_dir = f"/run/user/{user_uid}"
            if Path(runtime_dir).exists():
                env["XDG_RUNTIME_DIR"] = runtime_dir
                pulse_path = f"{runtime_dir}/pulse"
                if Path(pulse_path).exists():
                    env["PULSE_RUNTIME_PATH"] = pulse_path
                pipewire_path = f"{runtime_dir}/pipewire-0"
                if Path(pipewire_path).exists():
                    env["PIPEWIRE_RUNTIME_DIR"] = runtime_dir
        except Exception as e:
            print(f"[TTS] Could not set up user audio env: {e}")

    played = False

    if not played:
        try:
            subprocess.run(["pw-play", file_path], timeout=120, check=True,
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            played = True
            print("[TTS] Playback via PipeWire")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if not played:
        try:
            subprocess.run(["paplay", file_path], timeout=120, check=True,
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            played = True
            print("[TTS] Playback via PulseAudio")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if not played:
        try:
            subprocess.run(["aplay", "-q", file_path], timeout=120, check=True,
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            played = True
            print("[TTS] Playback via ALSA")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if not played:
        print("[TTS] ERROR: All audio backends failed")


def play_tts_audio(wav_path: str, output_device: Optional[str] = None):
    """Play a synthesised TTS WAV file (blocking) then delete it."""
    if not wav_path or wav_path == "__played__":
        return
    try:
        if not Path(wav_path).exists():
            print(f"[TTS] Audio file not found: {wav_path}")
            return
        if _tts_stop_flag.is_set():
            return
        if output_device is None:
            output_device = getattr(cfg, "SOUND_OUTPUT_DEVICE", "Default Sound Device")
        if output_device == "Default Sound Device":
            output_device = "default"
        print(f"[TTS] Playing audio: {wav_path}")
        _play_audio_file(wav_path, output_device)
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
    """Stop any ongoing TTS playback."""
    global _tts_thread
    _tts_stop_flag.set()
    if _tts_thread and _tts_thread.is_alive():
        _tts_thread.join(timeout=1.0)


def clear_tts_stop():
    """Clear the TTS stop flag so a new playback can start."""
    _tts_stop_flag.clear()


def is_speaking() -> bool:
    """Return True if TTS is actively playing."""
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