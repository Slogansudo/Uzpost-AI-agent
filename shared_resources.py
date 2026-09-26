"""
shared_resources.py — Markaziy resurslar
=========================================
Barcha komponentlar (faq, tracking, location, calculator) shu moduldan
bitta shared model va HTTP session ishlatadi.

MUAMMO (avval):
  - search_service.py  → o'z modeli (modul import da yuklanadi, blok qiladi)
  - tracking/rag.py    → o'z modeli (xotirada 2x E5-large = ~2.6 GB)
  - location/handler.py → har request da yangi ClientSession
  - tracking/rag.py    → har request da yangi ClientSession
  - calculator         → har request da yangi ClientSession

YECHIM (endi):
  - shared_resources.py → bitta model (1.3 GB) + bitta HTTP session
  - main.py startup → await init_all() bir marta
  - Barcha komponentlar: from shared_resources import get_model, get_http
  - Xotira: ~1.3 GB tejash, latency: ~0 (model allaqachon xotirada)

Ishlatish:
    # main.py startup
    from shared_resources import init_all, close_all
    await init_all()

    # istalgan modul
    from shared_resources import get_model, get_http
    model = get_model()   # SentenceTransformer (blok qilmaydi)
    http  = get_http()    # aiohttp.ClientSession (blok qilmaydi)
"""

from __future__ import annotations
import asyncio
import os
from typing import Optional
from FlagEmbedding import BGEM3FlagModel
import aiohttp

# ── Model nomi ────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-m3"

# ── Global singletonlar ─────────────────────────────
_model:        Optional[BGEM3FlagModel]    = None
_http_session: Optional[aiohttp.ClientSession] = None
_lock          = asyncio.Lock()
_initialized   = False


# ── Init / Close ──────────────────────────────────────────────────────────────

async def init_all() -> None:
    """
    main.py @startup da BIR MARTA chaqiriladi.
    Thread pool da model yuklanadi (async blok qilmaydi).
    """
    global _model, _http_session, _initialized

    async with _lock:
        if _initialized:
            return

        loop = asyncio.get_event_loop()

        # 1. Model (blocking, thread pool da)
        if _model is None:
            _model = await loop.run_in_executor(
                None,
                lambda: BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            )
            print("[SharedResources] bge-m3 tayyor ✓")

        # 2. HTTP session (async, bir marta)
        if _http_session is None or _http_session.closed:
            connector     = aiohttp.TCPConnector(
                limit=50,                # max parallel connections
                limit_per_host=20,
                keepalive_timeout=120,
                enable_cleanup_closed=True,
            )
            _http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30),
            )
            print("[SharedResources] HTTP session tayyor ✓")

        _initialized = True
        print("[SharedResources] Barcha resurslar tayyor ✓")


async def close_all() -> None:
    """main.py @shutdown da chaqiriladi."""
    global _http_session, _initialized

    if _http_session and not _http_session.closed:
        await _http_session.close()
        print("[SharedResources] HTTP session yopildi ✓")

    _initialized = False


# ── Getterlar (barcha modullar shu orqali oladi) ──────────────────────────────

def get_model() -> BGEM3FlagModel:
    if _model is None:
        raise RuntimeError("BGE-M3 yuklanmagan — init_all() chaqirilganmi?")
    return _model


def get_http() -> aiohttp.ClientSession:
    """
    Sinxron getter — init_all() dan keyin xavfsiz.
    Yangi session yaratmaydi — shared singletonni qaytaradi.
    """
    if _http_session is None or _http_session.closed:
        raise RuntimeError(
            "HTTP session tayyor emas — init_all() chaqirilganmi?"
        )
    return _http_session
