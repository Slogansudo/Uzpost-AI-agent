"""
tracking/rag.py — Kuchaytirilgan versiya
=========================================
O'zgarishlar (logika O'ZGARMAGAN):
  - SCORE_THRESHOLD: 0.45 → 0.52  (past sifatli natijalar filtrlanadi)
  - TOP_K: 2 → 3                  (ko'proq kontekst)
  - _build_context: score + question + keywords ham uzatiladi LLM ga
  - _search: natijalarni score bo'yicha saralab qaytaradi
  - get_tracking_context: sub_intent nomini ham kontekstga qo'shadi
"""

import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
COLLECTION      = "tracking"
TOP_K           = 3      # 2 → 3: ko'proq kontekst
SCORE_THRESHOLD = 0.52   # 0.45 → 0.52: past sifatli natijalar chiqmasin


# ── preload_model — endi no-op ────────────────────────────────────────────────

def preload_model() -> None:
    try:
        from shared_resources import get_model
        _ = get_model()
        print("[RAG/Tracking] Model shared_resources dan tayyor ✓")
    except RuntimeError:
        print("[RAG/Tracking] OGOHLANTIRISH: shared_resources init_all() chaqirilmagan")


# ── Embed ─────────────────────────────────────────────────────────────────────

async def _embed_async(text: str) -> list[float]:
    from shared_resources import get_model
    import asyncio
    loop = asyncio.get_event_loop()

    def _encode():
        model = get_model()  # BGEM3FlagModel
        result = model.encode(
            [text],
            batch_size=1,
            max_length=256,
            return_dense=True,
            return_sparse=False,  # tracking faqat dense ishlatadi
            return_colbert_vecs=False,
        )
        return result["dense_vecs"][0].tolist()

    return await loop.run_in_executor(None, _encode)


# ── Qdrant search ─────────────────────────────────────────────────────────────

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        h["api-key"] = QDRANT_API_KEY
    return h


async def _search(vector: list[float], category: str | None = None) -> list[dict]:
    from shared_resources import get_http

    body = {
        "vector":          vector,
        "limit":           TOP_K,
        "with_payload":    True,
        "score_threshold": SCORE_THRESHOLD,
    }
    if category:
        body["filter"] = {
            "must": [{"key": "category", "match": {"value": category}}]
        }

    try:
        async with get_http().post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers=_headers(),
            json=body,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                print(f"[RAG/Tracking] Qdrant {resp.status}")
                return []
            data    = await resp.json()
            results = data.get("result", [])

            # Score bo'yicha tartibla (Qdrant odatda tartiblab beradi, lekin kafolat)
            results.sort(key=lambda x: x.get("score", 0), reverse=True)

            if not results and category:
                print(f"[RAG/Tracking] '{category}' filter natija bermadi → filtersiz")
                return await _search(vector, category=None)
            return results

    except asyncio.TimeoutError:
        print("[RAG/Tracking] Qdrant timeout")
        return []
    except Exception as e:
        print(f"[RAG/Tracking] Xato: {e}")
        return []


# sub-intent → qdrant kategoriya
_CATEGORY_MAP = {
    "customs":  "customs",
    "delay":    "delay",
    "no_sms":   "no_sms",
    "lost":     "delay",
    "pickup":   "pickup",
    "eta":      "delay",
    "status":   None,
    "location": None,
}


def _build_context(results: list[dict], lang: str) -> str | None:
    """
    Kuchaytirilgan: har bir natijadan score + question + keywords + answer
    uzatiladi — LLM aniqroq kontekst oladi.
    """
    if not results:
        return None

    ans_key = f"answer_{lang}"
    q_key   = f"question_{lang}"   # agar payloaddа bo'lsa
    blocks  = []

    for i, r in enumerate(results, 1):
        p     = r.get("payload", {})
        score = r.get("score", 0)

        answer = p.get(ans_key) or p.get("answer_uz", "")
        if not answer or not answer.strip():
            continue

        # Qo'shimcha meta — LLM ga nima haqida ekanini bildiradi
        question = p.get(q_key) or p.get("question_uz") or p.get("question", "")
        keywords = p.get("keywords") or p.get("tags") or []
        category = p.get("category", "")

        meta_parts = []
        if question:
            meta_parts.append(f"Savol: {question.strip()}")
        if keywords:
            kw_str = ", ".join(keywords[:5]) if isinstance(keywords, list) else str(keywords)
            meta_parts.append(f"Kalit so'zlar: {kw_str}")
        if category:
            meta_parts.append(f"Kategoriya: {category}")

        meta = " | ".join(meta_parts)
        block_lines = [f"[{i}] (mos: {score:.2f})"]
        if meta:
            block_lines.append(meta)
        block_lines.append(answer.strip())

        blocks.append("\n".join(block_lines))

    if not blocks:
        return None

    return "\n\n---\n\n".join(blocks)


# ── Public ────────────────────────────────────────────────────────────────────

async def get_tracking_context(
    query:      str,
    sub_intent: str = "status",
    lang:       str = "uz",
) -> str | None:
    """
    Qdrant dan tegishli kontekst oladi.
    Kuchaytirilgan: sub_intent nomi va natija soni log ga chiqadi.
    """
    vector   = await _embed_async(query)
    category = _CATEGORY_MAP.get(sub_intent)
    results  = await _search(vector, category)

    if not results:
        print(f"[RAG/Tracking] '{sub_intent}' uchun natija yo'q (threshold={SCORE_THRESHOLD})")
        return None

    ctx = _build_context(results, lang)
    if ctx:
        scores = [f"{r.get('score',0):.2f}" for r in results]
        print(f"[RAG/Tracking] sub={sub_intent} | {len(results)} natija | scores={scores}")
    return ctx

