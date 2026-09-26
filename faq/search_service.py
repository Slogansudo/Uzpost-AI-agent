"""
faq/search_service.py  —  v6.1: BGE-M3 + SITUATION_COLLECTION
==============================================================================

v6.0 dan farqi (bugfix):
  1. SPARSE QUERY FORMAT TO'G'RILANDI
     Eski (noto'g'ri, HTTP 400 sababi):
       "query": {"sparse": {"indices": [...], "values": [...]}}

     To'g'ri (Qdrant docs bo'yicha):
       "query": {"indices": [...], "values": [...]}
       "using": "sparse"

  2. QDRANT VERSION GUARD
     prefetch+fusion ishlamasa → avtomatik dense-only fallback
     → Qdrant < 1.10 da ham ishlaydi

  3. SITUATION-FIRST PIPELINE — categoriyadan KO'RA USTUN
     Situation topilsa → to'g'ri javob, category pipeline o'tkazib yuboriladi
     SITUATION_MIN_SCORE = 0.72 (0.78 dan pasaytirildi — ko'proq hit)

  4. DENSE-ONLY FALLBACK for situation search
     Sparse format xato bo'lsa ham dense bilan ishlaydi

O'ZGARMAGAN:
  - search() public API
  - SearchService, init_resources(), close_resources()
  - BM25 logikasi
  - _build_context(), _extract_relevant_sections()
  - LLM system promptlar
  - show_context(), interactive_context_test()
"""
from __future__ import annotations

import os
import re
import pickle
import asyncio
import aiohttp
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .search_synonim import (
    _SYNONYMS,
    _CATEGORY_KEYWORDS,
    _CATEGORY_TO_SERVICE,
    _COMPARISON_PAIRS,
    _INTENT_DEFAULT_SERVICES,
    _CATEGORY_EXCLUSIONS,
    _INTENT_PATTERNS,
    _INTENT_TO_CHUNK_INTENTS,
    _LOW_PRIORITY_CATEGORIES,
)

try:
    from lingua import Language, LanguageDetectorBuilder
    _detector = LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.RUSSIAN
    ).build()
    LINGUA_OK = True
except ImportError:
    LINGUA_OK = False

load_dotenv()

# ── Konfiguratsiya ─────────────────────────────────────────────────────────────

QDRANT_URL           = os.getenv("QDRANT_URL",             "http://localhost:6333")
QDRANT_KEY           = os.getenv("QDRANT_API_KEY",         "")
CHUNK_COLLECTION     = os.getenv("CHUNK_COLLECTION",       "uzpost_chunks_v1")
SERVICE_COLLECTION   = os.getenv("SERVICE_COLLECTION",     "uzpost_services_v1")
CATEGORY_COLLECTION  = os.getenv("CATEGORY_COLLECTION",    "uzpost_categories_v1")
SITUATION_COLLECTION = os.getenv("SITUATION_COLLECTION",   "uzpost_situations_v1")
GROQ_KEY             = os.getenv("GROQ_API_KEY",           "")
BM25_PATH            = Path(os.getenv("BM25_INDEX_PATH",   "faq/bm25_index.pkl"))
GROQ_MODEL           = os.getenv("GROQ_MODEL_PREMIUM",     "llama3-70b-versatile")

# ── Hybrid search parametrlari ─────────────────────────────────────────────────
HYBRID_TOP_K         = 8
HYBRID_FINAL_TOP_K   = 4

# ── SITUATION parametrlari ─────────────────────────────────────────────────────
# 0.72 — pasaytirildi (0.78 dan): ko'proq situation hit bo'ladi
# Agar noto'g'ri javoblar chiqsa → 0.75-0.80 ga ko'taring
SITUATION_MIN_SCORE  = 0.72
SITUATION_TOP_K      = 4
SITUATION_CONTEXT_K  = 2

# ── Classify parametrlari ──────────────────────────────────────────────────────
CLASSIFY_TOP_K       = 5
CLASSIFY_AMBIGUITY   = 0.02

# ── ColBERT ────────────────────────────────────────────────────────────────────
COLBERT_RERANK       = True
COLBERT_TOP_N        = 8

# ── BM25 / Service ────────────────────────────────────────────────────────────
TOP_BM25             = 7
TOP_CHUNKS           = 2
TOP_SERVICES         = 2
FOLLOWUP_MAX_TOKENS  = 4

# ── Qdrant versiya flag (runtime da aniqlanadi) ────────────────────────────────
# True  → prefetch+fusion (Qdrant ≥ 1.10)
# False → oddiy /search endpoint (Qdrant < 1.10 yoki xato bo'lsa)
_QDRANT_SUPPORTS_QUERY = True   # birinchi muvaffaqiyatli so'rovdan keyin True saqlanadi

# ── BM25 yuklash ──────────────────────────────────────────────────────────────

_BM25_DATA: Optional[dict] = None
if BM25_PATH.exists():
    with open(BM25_PATH, "rb") as _f:
        _BM25_DATA = pickle.load(_f)
    print(f"[SearchService] BM25 tayyor: {len(_BM25_DATA['docs'])} doc ✓")
else:
    print(f"[SearchService] BM25 topilmadi: {BM25_PATH}")

_LLM: Optional[ChatGroq] = None
_HTTP_SESSION: Optional[aiohttp.ClientSession] = None
_BGE_MODEL_STANDALONE = None


# ── BGE-M3 model olish ────────────────────────────────────────────────────────

def _get_bge_model():
    try:
        from shared_resources import get_model
        return get_model()          # ← get_model() funksiyasi orqali
    except (ImportError, RuntimeError):
        return _BGE_MODEL_STANDALONE


def _embed_query_all(text: str) -> tuple[list[float], dict, object]:
    """
    BGE-M3: dense + sparse + colbert — bitta forward pass.

    Sparse format qaytaradi:
      {"indices": [int, ...], "values": [float, ...]}
    Bu Qdrant /points/query uchun to'g'ridan ishlatiladi.
    """
    model = _get_bge_model()

    if model is None:
        # shared_resources orqali dense olishga urinish
        try:
            from shared_resources import embed
            dense = embed(text, is_query=True)
            return dense, {}, None
        except (ImportError, RuntimeError):
            raise RuntimeError("BGE-M3 modeli yuklanmagan")

    result = model.encode(
        [text],
        batch_size=1,
        max_length=256,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=COLBERT_RERANK,
    )

    # Dense vector
    dense = result["dense_vecs"][0].tolist()

    # Sparse → Qdrant formati: {"indices": [...], "values": [...]}
    # MUHIM: token_id int bo'lishi shart (BGE-M3 string qaytarishi mumkin)
    lw = result["lexical_weights"][0]
    indices: list[int]   = []
    values:  list[float] = []
    for token_id, weight in lw.items():
        indices.append(int(token_id))
        values.append(float(weight))
    sparse = {"indices": indices, "values": values}

    # ColBERT vecs (ixtiyoriy)
    colbert_vecs = None
    if COLBERT_RERANK:
        raw = result.get("colbert_vecs")
        if raw is not None and len(raw) > 0:
            colbert_vecs = raw[0]

    return dense, sparse, colbert_vecs


# ── Resurslar ──────────────────────────────────────────────────────────────────

async def init_resources() -> None:
    global _LLM, _HTTP_SESSION

    if GROQ_KEY and _LLM is None:
        _LLM = ChatGroq(model=GROQ_MODEL, temperature=0.0, api_key=GROQ_KEY)
        print("[SearchService] LLM tayyor ✓")

    if _HTTP_SESSION is None or _HTTP_SESSION.closed:
        _HTTP_SESSION = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        print("[SearchService] HTTP session tayyor ✓")

    print("[SearchService] init_resources() bajarildi ✓")


async def close_resources() -> None:
    global _HTTP_SESSION
    if _HTTP_SESSION and not _HTTP_SESSION.closed:
        await _HTTP_SESSION.close()
        print("[SearchService] HTTP session yopildi ✓")


def _get_http() -> aiohttp.ClientSession:
    try:
        from shared_resources import get_http
        return get_http()
    except (ImportError, RuntimeError):
        pass
    if _HTTP_SESSION and not _HTTP_SESSION.closed:
        return _HTTP_SESSION
    raise RuntimeError("HTTP session tayyor emas")


# ── Stopwords & tokenizer ──────────────────────────────────────────────────────

_STOPWORDS = {
    "va","yoki","ham","bu","u","uchun","bilan","dan","ga","da","ni","ning",
    "boladi","mumkin","kerak","bolgan","qilish","bir","shu","ular","men",
    "sen","biz","ozi","qiladi","qiling","lekin","agar","hatto","faqat",
    "esa","chunki","i","v","na","po","chto","kak","ne","eto","ot","do",
    "a","no","ili","za","iz","pri","k","s","dlya","o","the","an","is",
    "in","to","of","or","and","for","be","it","at","by","we","as","if",
    "on","are","can","not","this","that","was","has","have","will","with", "bormi","qanaqa","qanday","nima","qaysi","qachon",
    "где","когда","какой","какая","можно", "what","when","where","which","how"
}


def _tokenize(text: str) -> list[str]:
    text = re.sub(r"[ʻʼ`ʹʾ\u2018\u2019]", "'", text.lower())
    return [t for t in re.findall(r"[a-zA-Zа-яёА-ЯЁ']+", text)
            if t not in _STOPWORDS and len(t) > 1]


def _normalize(query: str) -> str:
    q = re.sub(r"[ʻʼ`ʹʾ\u2018\u2019]", "'", query.lower().strip())
    return " ".join(re.sub(r"[!?.,;:«»\"\u201c\u201d]", "", q).split())


_CONTEXT_BUDGET: dict[str, int] = {
    "price":1400,"storage":1000,"tracking":1000,"coverage":1000,
    "restriction":1000,"comparison":1100,"complaint":1000,"info":1100,
}
_LLM_MAX_TOKENS: dict[str, int] = {
    "price":500,"storage":450,"comparison":500,"info":500,
    "restriction":400,"tracking":350,"coverage":450,"complaint":350,
}

# ── System promptlar ───────────────────────────────────────────────────────────

_SYSTEM_PROMPTS: dict[str, str] = {
    "uz": (
        "Sen Pochtachi Bek — UzPost rasmiy yordamchi botisman. "
        "Tajribali, samimiy va proaktiv operator sifatida gapir.\n\n"

        "JAVOB TARTIBI:\n"
        "1. Savolga to'liq, aniq va batafsil javob ber (odatda 4-6 gap).\n"
        "2. Agar ma'lumot kontekstda bo'lsa — undan foydalanish shart, raqamlarni o'zgartirma.\n"
        "3. Follow-up savol HAR DOIM EMAS — faqat mijozga haqiqatan yordam bersa qo'sh.\n"
        "   - Follow-up MIJOZNING AYNAN SAVOLIDAN kelib chiqsin, shablon bo'lmasin.\n"
        "   - Misol: mijoz 'chet elga Bir qadam bilan yuborsa bo'ladimi' desa →\n"
        "     'Bir qadam faqat O'zbekiston ichida ishlaydi. Chet elga bo'lsa EMS yoki\n"
        "      xalqaro posilka bor — qaysi davlatga, qaysi hududdan jo'natmoqchisiz?\n"
        "      Shunga qarab aniq tarif va narxni ayta olaman.' kabi — savoldagi muammoni hal qiluvchi.\n"
        "   - Mijoz aniq, yopiq savol bergan va javob to'liq bo'lsa (masalan 'rahmat',\n"
        "     bitta faktli savol) — follow-up shart emas, ortiqcha savol berma.\n\n"

        "USLUB:\n"
        "- Tabiiy, samimiy va insoniy yoz — robot kabi emas.\n"
        "- 4-6 gap. Gap ko'rinishida, ro'yxat shaklida emas.\n"
        "- Faqat UzPost xizmatlari haqida.\n"
        "- Noto'g'ri ma'lumot uydirma.\n\n"

        "TAQIQLAR:\n"
        "- Emoji va markdown teglar ishlatma.\n"
        "- 'Boshqa savolingiz bo'lsa' kabi generik iboralar ishlatma.\n"
        "- Har javob oxirida bir xil qotib qolgan savolni takrorlama.\n"
        "- Tariflar va sanalarni o'ylab topma.\n"
        "- 1165 yoki uz.post FAQAT aniq narx, manzil, ish vaqti ma'lumoti\n"
        "  bo'lmaganida ayt — har javobda emas!"
    ),

    "ru": (
        "Ты Pochtachi Bek — официальный бот-помощник UzPost. "
        "Говори как опытный, внимательный и проактивный оператор.\n\n"

        "ПОРЯДОК ОТВЕТА:\n"
        "1. Ответь полно и подробно на вопрос (обычно 4-6 предложений).\n"
        "2. Если в контексте есть данные — используй их, не меняй цифры.\n"
        "3. Follow-up вопрос НЕ ВСЕГДА — только если он реально помогает клиенту.\n"
        "   - Follow-up должен вытекать ИЗ КОНКРЕТНОГО вопроса клиента, не шаблон.\n"
        "   - Пример: клиент спрашивает 'можно ли отправить за границу через Bir qadam' →\n"
        "     'Bir qadam работает только внутри Узбекистана. За границу есть EMS или\n"
        "      международная посылка — в какую страну и из какого региона отправляете?\n"
        "      Тогда подскажу точный тариф.' — решающий проблему из вопроса.\n"
        "   - Если вопрос закрытый и ответ полный ('спасибо', один факт) — без follow-up.\n\n"

        "СТИЛЬ:\n"
        "- Естественно и по-человечески — не как робот.\n"
        "- 4-6 предложений. Текстом, не списком.\n"
        "- Только об UzPost.\n"
        "- Не придумывай информацию.\n\n"

        "ЗАПРЕЩЕНО:\n"
        "- Emoji и markdown.\n"
        "- Шаблонные фразы 'Если есть вопросы — обращайтесь'.\n"
        "- Повторять один и тот же шаблонный вопрос в конце каждого ответа.\n"
        "- Придумывать тарифы и даты.\n"
        "- 1165 и uz.post — ТОЛЬКО когда нет точных данных о тарифах,\n"
        "  адресах, часах работы — не в каждом ответе!"
    ),

    "en": (
        "You are Pochtachi Bek — official UzPost assistant bot. "
        "Be experienced, warm, and proactive.\n\n"

        "RESPONSE STRUCTURE:\n"
        "1. Answer the question fully and in detail (usually 4-6 sentences).\n"
        "2. If the context has data — use it, don't change numbers.\n"
        "3. A follow-up is NOT ALWAYS needed — add one only if it truly helps.\n"
        "   - The follow-up must follow FROM THE CUSTOMER'S SPECIFIC question, not a template.\n"
        "   - Example: customer asks 'can I ship abroad with Bir qadam' →\n"
        "     'Bir qadam works only within Uzbekistan. For abroad there's EMS or\n"
        "      international parcel — which country and from which region are you shipping?\n"
        "      Then I can give you the exact tariff.' — solving the problem in the question.\n"
        "   - If the question is closed and the answer is complete ('thanks', a single fact) — no follow-up.\n\n"

        "STYLE:\n"
        "- Natural and human — not like a robot.\n"
        "- 4-6 sentences. Prose, not bullet points.\n"
        "- UzPost only.\n"
        "- Never invent information.\n\n"

        "PROHIBITED:\n"
        "- Emoji and markdown.\n"
        "- Generic phrases like 'Feel free to ask anytime'.\n"
        "- Repeating the same templated question at the end of every answer.\n"
        "- Inventing rates or dates.\n"
        "- Mention 1165 or uz.post ONLY when specific data (rates, addresses,\n"
        "  hours) is unavailable — not in every response!"
    ),
}


_SITUATION_SYSTEM_PROMPTS: dict[str, str] = {
    "uz": (
        "Sen Pochtachi Bek — UzPost rasmiy yordamchi botisman. "
        "Faqat o'zbek tilida javob ber.\n\n"

        "Quyida mijoz savoliga mos holatlar va javoblar berilgan. "
        "Eng mosini tanlab, o'z so'zlaring bilan tabiiy va qisqa javob ber.\n\n"

        "QOIDALAR:\n"
        "- Savolga to'g'ridan javob ber.\n"
        "- Holatdagi raqam, muddat va qoidalarni o'zgartirma.\n"
        "- Faqat UzPost haqida gapir.\n"
        "- Boshqa xizmatlarni tilga olma.\n\n"

        "USLUB:\n"
        "- Qisqa, aniq va insoniy yoz.\n"
        "- Bir nechta holat mos bo'lsa, eng muhimini tanla.\n"
        "- Emoji, markdown va texnik teglar ishlatma.\n"
        "- Ro'yxatsiz oddiy gap bilan yoz.\n"
        "- Javob oxirida bitta proaktiv taklif qo'sh."
    ),

    "ru": (
        "Ты Pochtachi Bek — официальный бот-помощник UzPost. "
        "Отвечай только на русском языке.\n\n"

        "Ниже есть ситуации и готовые ответы. "
        "Выбери наиболее подходящий и ответь своими словами — кратко и естественно.\n\n"

        "ПРАВИЛА:\n"
        "- Отвечай прямо на вопрос.\n"
        "- Не изменяй цифры, сроки и правила из ситуации.\n"
        "- Говори только об UzPost.\n"
        "- Не упоминай другие службы.\n\n"

        "СТИЛЬ:\n"
        "- Кратко, понятно и по-человечески.\n"
        "- Если подходит несколько ситуаций — выбери главную.\n"
        "- Без emoji, markdown и техтегов.\n"
        "- Не списком.\n"
        "- В конце добавь один проактивный вопрос или предложение."
    ),

    "en": (
        "You are Pochtachi Bek — official UzPost assistant bot. "
        "Reply only in English.\n\n"

        "Below are situations and prepared answers matching the customer's question. "
        "Choose the most relevant one and answer naturally in your own words.\n\n"

        "RULES:\n"
        "- Answer directly.\n"
        "- Do not change numbers, deadlines, or rules from the situation.\n"
        "- Talk only about UzPost.\n"
        "- Do not mention other services.\n\n"

        "STYLE:\n"
        "- Brief, clear, and human.\n"
        "- If several situations fit, choose the most relevant.\n"
        "- No emoji, markdown, or technical tags.\n"
        "- No list formatting.\n"
        "- End with one proactive question or offer."
    ),
}

_HINTS: dict[str, dict[str, str]] = {
    "uz": {
        "tracking": "Trek raqamingizni yuboring, kuzatib beray.",
        "complaint": "Muammo davom etsa: 1165 | info@pochta.uz",
        "storage":   "Aniq saqlash muddati uchun 1165 ga murojaat qiling.",
    },
    "ru": {
        "tracking": "Отправьте трек-номер — проверю статус.",
        "complaint": "Проблема продолжается: 1165 | info@pochta.uz",
        "storage":   "Уточните точный срок хранения по номеру 1165.",
    },
    "en": {
        "tracking": "Send your tracking number and I'll check the status.",
        "complaint": "Issue persists: 1165 | info@pochta.uz",
        "storage":   "Call 1165 for exact storage terms.",
    },
}

_NO_DATA: dict[str, str] = {
    "uz": "Bu bo'yicha aniq ma'lumotim yo'q. Batafsil uchun 1165 ga murojaat qiling yoki uz.post saytini tekshiring.",
    "ru": "Точной информации по этому вопросу у меня нет. Уточните по номеру 1165 или на сайте uz.post.",
    "en": "I don't have exact information on this. Please call 1165 or check uz.post for details.",
}

# ── Til aniqlash ───────────────────────────────────────────────────────────────

_CYRILLIC    = re.compile(r"[а-яёА-ЯЁ]")
_UZ_SPECIFIC = re.compile(
    r"(o'[a-z]|g'[a-z]|o`[a-z]|g`[a-z]"
    r"|\b\w{3,}(da|ga|ni|dan|dagi|ning|lar|mi|gina|kina|qina)\b)", re.I,
)
_UZ_CORE = re.compile(
    r"\b(salom|nima|qani|qachon|qayerda|necha|bormi|mavjudmi|kerak|"
    r"pochta|xizmat|haqida|toshkent|samarqand|namangan|andijon|buxoro|"
    r"termiz|nukus|yuborish|jonatish|yetkazish|narxi|qancha|tarif|"
    r"posilka|banderol|xat|kuryer|bir|qadam|ekspress|gibrid|gibrit|"
    r"tanishtir|tushuntir|malumot|batafsil|qanday|qilib|qilaman)\b", re.I,
)
_EN_MARKERS = re.compile(
    r"\b(is|are|there|where|what|which|how|the|service|price|cost|"
    r"delivery|parcel|letter|send|post|please|tell|explain|can|does|"
    r"do|will|information|about)\b", re.I,
)


def _detect_lang(text: str) -> str:
    text = text.strip()
    if not text:
        return "uz"
    if _CYRILLIC.search(text):
        return "ru"
    en_count = len(_EN_MARKERS.findall(text))
    if en_count >= 2:
        if LINGUA_OK:
            try:
                if _detector.detect_language_of(text) == Language.ENGLISH:
                    return "en"
            except Exception:
                pass
        if en_count >= 3:
            return "en"
    if _UZ_SPECIFIC.search(text) or _UZ_CORE.search(text):
        return "uz"
    if LINGUA_OK:
        try:
            r = _detector.detect_language_of(text)
            if r == Language.ENGLISH:
                return "en"
            if r == Language.RUSSIAN:
                return "ru"
        except Exception:
            pass
    return "uz"


def _detect_intent(text: str) -> str:
    t = text.lower()
    for intent, patterns in _INTENT_PATTERNS.items():
        if any(p in t for p in patterns):
            return intent
    return "info"


def _expand_query(query: str) -> tuple[str, Optional[str]]:
    q_lower = query.lower()
    tokens  = q_lower.split()
    expanded: list[str] = []

    for token in tokens:
        expanded.append(token)
        if token in _SYNONYMS:
            expanded.extend(_SYNONYMS[token])
        else:
            for key, syns in _SYNONYMS.items():
                if len(key) > 3 and token.startswith(key):
                    expanded.extend(syns)
                    break

    CATEGORY_ORDER = [
        "bir_qadam","hybrid_mail","cecogram","ems","m_bag",
        "courier","small_packet","wrapper","postcard",
        "money_transfer","fulfilment","storage","letter","parcel",
    ]
    detected = next(
        (c for c in CATEGORY_ORDER
         if any(kw in q_lower for kw in _CATEGORY_KEYWORDS.get(c, []))),
        None,
    )
    return " ".join(expanded), detected


# ── Multi-turn ─────────────────────────────────────────────────────────────────

def _resolve_category(
    detected: Optional[str],
    query: str,
    conv_state: Optional[dict],
) -> tuple[Optional[str], bool]:
    if not conv_state or conv_state.get("intent") != "faq":
        return detected, False
    prev = conv_state.get("detected_category")
    if detected:
        if detected != prev:
            print(f"[MultiTurn] Yangi category: {detected} (avval: {prev})")
        return detected, False
    if not prev:
        return None, False
    if len(_tokenize(query)) <= FOLLOWUP_MAX_TOKENS:
        print(f"[MultiTurn] Qisqa follow-up → prev_cat={prev}")
        return prev, True
    print("[MultiTurn] Uzun savol, category reset")
    return None, False


# ── Qdrant headers ─────────────────────────────────────────────────────────────

def _qdrant_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if QDRANT_KEY:
        h["api-key"] = QDRANT_KEY
    return h


# ═══════════════════════════════════════════════════════════════════════════════
# QDRANT HYBRID SEARCH HELPER
# Sparse format bugfix + version fallback
# ═══════════════════════════════════════════════════════════════════════════════

async def _qdrant_hybrid_search(
    collection:  str,
    dense_vec:   list[float],
    sparse_vec:  dict,
    lang:        str,
    top_k:       int,
    extra_filter: Optional[dict] = None,
    timeout:     float = 4.0,
) -> list[dict]:
    """
    Universal hybrid search helper.

    TO'G'RI SPARSE FORMAT (v6.0 bugfix):
      prefetch[1]["query"] = {"indices": [...], "values": [...]}  ← to'g'ri
      EMAS: {"sparse": {"indices": [...], "values": [...]}}       ← noto'g'ri

    Qdrant ≥ 1.10 → /points/query + prefetch + fusion: rrf
    Qdrant < 1.10  → /points/search (dense only, sparse o'tkazib yuboriladi)

    extra_filter: {"must": [...]} formatida qo'shimcha filter.
    """
    global _QDRANT_SUPPORTS_QUERY

    session = _get_http()
    headers = _qdrant_headers()

    lang_filter: dict = {"must": [{"key": "lang", "match": {"value": lang}}]}
    if extra_filter:
        lang_filter["must"].extend(extra_filter.get("must", []))

    # ── Prefetch + RRF (Qdrant ≥ 1.10) ──────────────────────────────────────
    if _QDRANT_SUPPORTS_QUERY:
        prefetch: list[dict] = [
            {
                "query":  dense_vec,   # dense: list sifatida beriladi
                "using":  "dense",
                "limit":  top_k,
                "filter": lang_filter,
            },
        ]

        # Sparse faqat indices bo'lsa qo'shamiz
        if sparse_vec.get("indices"):
            prefetch.append({
                # TO'G'RI FORMAT: "query" = sparse vector object
                "query": {
                    "indices": sparse_vec["indices"],
                    "values":  sparse_vec["values"],
                },
                "using":  "sparse",
                "limit":  top_k,
                "filter": lang_filter,
            })

        query_payload = {
            "prefetch":     prefetch,
            "query":        {"fusion": "rrf"},
            "limit":        top_k,
            "with_payload": True,
            "with_vector":  False,
        }

        try:
            async with session.post(
                f"{QDRANT_URL}/collections/{collection}/points/query",
                headers=headers,
                json=query_payload,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    data   = await resp.json()
                    # Qdrant 1.10+ → result.points; ba'zi versiyalar → result[]
                    raw = data.get("result", {})
                    points = raw.get("points", raw) if isinstance(raw, dict) else raw
                    return points if isinstance(points, list) else []

                elif resp.status == 400:
                    # Versiya muammosi yoki format xato — dense-only ga tush
                    err_body = await resp.text()
                    print(
                        f"[HybridSearch] HTTP 400 ({collection}) → "
                        f"dense-only fallback. Error: {err_body[:200]}"
                    )
                    _QDRANT_SUPPORTS_QUERY = False
                    # dense-only ga o'tadi (quyida)

                else:
                    print(f"[HybridSearch] HTTP {resp.status} ({collection})")
                    return []

        except asyncio.TimeoutError:
            print(f"[HybridSearch] Timeout ({collection})")
            return []
        except aiohttp.ClientConnectorError:
            print(f"[HybridSearch] Qdrant ulanmadi ({QDRANT_URL})")
            return []
        except Exception as e:
            print(f"[HybridSearch] {type(e).__name__}: {e}")
            return []

    # ── Dense-only fallback (Qdrant < 1.10 yoki 400 xatosi) ─────────────────
    search_payload = {
        "vector": {"name": "dense", "vector": dense_vec},
        "limit":        top_k,
        "with_payload": True,
        "with_vector":  False,
        "filter":       lang_filter,
    }

    try:
        async with session.post(
            f"{QDRANT_URL}/collections/{collection}/points/search",
            headers=headers,
            json=search_payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("result", [])
            else:
                print(f"[HybridSearch dense] HTTP {resp.status} ({collection})")
                return []

    except Exception as e:
        print(f"[HybridSearch dense] {type(e).__name__}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SITUATION SEARCH — v6 asosiy pipeline bosqichi
# ═══════════════════════════════════════════════════════════════════════════════

async def _search_situations(
    query:      str,
    lang:       str,
    dense_vec:  list[float],
    sparse_vec: dict,
) -> list[dict]:
    """
    SITUATION_COLLECTION da hybrid qidiruv.

    Har bir situation → real muammo + tayyor javob juft.
    Score > SITUATION_MIN_SCORE bo'lsa → to'g'ri javob, category pipeline o'tkazib yuboriladi.

    Qaytaradi: [{"score": float, "payload": {...}}, ...]
    """
    points = await _qdrant_hybrid_search(
        collection=SITUATION_COLLECTION,
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        lang=lang,
        top_k=SITUATION_TOP_K,
        timeout=4.0,
    )

    results: list[dict] = []
    for pt in points:
        score = float(pt.get("score", 0))
        pl    = pt.get("payload", {})
        results.append({
            "score": score,
            "payload": {
                "situation_id":     pl.get("situation_id",     ""),
                "situation_type":   pl.get("situation_type",   ""),
                "related_services": pl.get("related_services", []),
                "category":         pl.get("category",         ""),
                "lang":             pl.get("lang",             lang),
                "question":         pl.get("question",         ""),
                "answer":           pl.get("answer",           ""),
                "tags":             pl.get("tags",             []),
                "weight":           float(pl.get("weight",     1.5)),
                "contact":          pl.get("contact",          {}),
                "service_name":     pl.get("service_name",     ""),
            },
        })

    if results:
        top = results[0]
        hit = "✓ HIT" if top["score"] >= SITUATION_MIN_SCORE else "✗ miss"
        print(
            f"[Situation] {hit} | "
            f"id={top['payload']['situation_id']} | "
            f"score={top['score']:.4f} | "
            f"threshold={SITUATION_MIN_SCORE}"
        )
    else:
        print("[Situation] Natija yo'q → fallback")

    return results


def _build_situation_context(
    situations: list[dict],
    max_k: int = SITUATION_CONTEXT_K,
) -> str:
    parts: list[str] = []
    for sit in situations[:max_k]:
        pl       = sit["payload"]
        question = pl.get("question", "")
        answer   = pl.get("answer",   "")
        if question and answer:
            parts.append(f"Holat: {question}\nJavob: {answer}")
    return "\n\n---\n\n".join(parts)


_SITUATION_HOOK: dict[str, dict[str, str]] = {
    "price_inquiry": {
        "uz": "Boshqa xizmat yoki yo'nalish narxini ham hisoblashimni xohlaysizmi?",
        "ru": "Хотите рассчитать стоимость для другой услуги или направления?",
        "en": "Would you like me to calculate rates for another service or destination?",
    },
    "price_calculation": {
        "uz": "Aniq og'irlik yoki o'lcham bo'yicha narx hisoblashimni xohlaysizmi?",
        "ru": "Хотите рассчитать точную стоимость по весу или размеру?",
        "en": "Would you like an exact calculation by weight or dimensions?",
    },
    "service_info": {
        "uz": "Bu xizmat bo'yicha jo'natish tartibi yoki hujjatlar haqida ham so'rasangiz bo'ladi.",
        "ru": "Можно спросить о порядке отправки или необходимых документах.",
        "en": "Feel free to ask about the sending process or required documents.",
    },
    "service_comparison": {
        "uz": "Boshqa xizmatlarni ham taqqoslashimni xohlaysizmi?",
        "ru": "Хотите сравнить другие услуги?",
        "en": "Would you like me to compare other services?",
    },
    "delivery_time": {
        "uz": "Boshqa shahar yoki mamlakat uchun yetkazish muddatini ham bilmoqchimisiz?",
        "ru": "Хотите узнать сроки доставки для другого города или страны?",
        "en": "Would you like to know delivery times for another city or country?",
    },
    "storage": {
        "uz": "Saqlash muddatini uzaytirish yoki to'lov haqida boshqa savol bormi?",
        "ru": "Есть вопросы о продлении срока хранения или оплате?",
        "en": "Any questions about extending storage or payment?",
    },
    "restriction": {
        "uz": "Boshqa buyum yoki mahsulot yuborish mumkinligi haqida ham so'rasangiz bo'ladi.",
        "ru": "Можно уточнить, можно ли отправить другие товары или предметы.",
        "en": "Feel free to ask whether other items can be shipped.",
    },
    "prohibited_items": {
        "uz": "Nima yuborish mumkin yoki mumkin emasligi haqida boshqa savol bormi?",
        "ru": "Есть вопросы о том, что можно или нельзя отправлять?",
        "en": "Any questions about what can or cannot be shipped?",
    },
    "complaint": {
        "uz": "Muammo davom etsa, 1165 yoki info@pochta.uz ga murojaat qiling.",
        "ru": "Если проблема продолжается — обратитесь на 1165 или info@pochta.uz.",
        "en": "If the issue continues, contact 1165 or info@pochta.uz.",
    },
    "compensation": {
        "uz": "Kompensatsiya arizasi berish tartibi haqida batafsil so'rasangiz bo'ladi.",
        "ru": "Можно спросить подробнее о порядке подачи заявления на компенсацию.",
        "en": "Feel free to ask more about the compensation claim process.",
    },
    "international": {
        "uz": "Xalqaro jo'natma bojxona yoki hujjatlari haqida boshqa savol bormi?",
        "ru": "Есть вопросы о таможне или документах для международных отправлений?",
        "en": "Any questions about customs or documents for international shipments?",
    },
    "customs": {
        "uz": "Bojxona to'lovi yoki kerakli hujjatlar haqida boshqa savol bormi?",
        "ru": "Есть вопросы по таможенным сборам или необходимым документам?",
        "en": "Any questions about customs fees or required documents?",
    },
    "default": {
        "uz": "Boshqa xizmat, narx yoki jo'natish haqida savolingiz bormi?",
        "ru": "Есть вопросы по другим услугам, тарифам или отправке?",
        "en": "Any questions about other services, rates, or shipping?",
    },
}

_TAG_TO_HOOK_TYPE: dict[str, str] = {
    "narx": "price_inquiry",
    "tarif": "price_inquiry",
    "price": "price_inquiry",
    "стоимость": "price_inquiry",
    "hisoblash": "price_calculation",
    "calculation": "price_calculation",
    "muddat": "delivery_time",
    "срок": "delivery_time",
    "delivery_time": "delivery_time",
    "saqlash": "storage",
    "storage": "storage",
    "хранение": "storage",
    "taqiq": "prohibited_items",
    "prohibited": "prohibited_items",
    "cheklov": "restriction",
    "restriction": "restriction",
    "shikoyat": "complaint",
    "complaint": "complaint",
    "kompensatsiya": "compensation",
    "bojxona": "customs",
    "customs": "customs",
    "таможня": "customs",
    "xalqaro": "international",
    "international": "international",
    "taqqoslash": "service_comparison",
    "comparison": "service_comparison",
}

def _build_hook_from_situation(
    situations: list[dict],
    lang:       str,
) -> str | None:
    if not situations:
        return None

    top = situations[0]["payload"]

    # 1. situation_type dan to'g'ridan
    sit_type = top.get("situation_type", "")
    if sit_type and sit_type in _SITUATION_HOOK:
        hook = _SITUATION_HOOK[sit_type].get(lang) or _SITUATION_HOOK[sit_type].get("uz", "")
        if hook:
            return hook

    # 2. Tags dan
    tags = top.get("tags", [])
    if isinstance(tags, list):
        for tag in tags:
            hook_type = _TAG_TO_HOOK_TYPE.get(str(tag).lower())
            if hook_type:
                hook = _SITUATION_HOOK[hook_type].get(lang) or _SITUATION_HOOK[hook_type].get("uz", "")
                if hook:
                    return hook

    # 3. Default fallback
    return _SITUATION_HOOK["default"].get(lang) or _SITUATION_HOOK["default"].get("uz", "")



async def _llm_answer_from_situations(
    query:      str,
    situations: list[dict],
    lang:       str,
    intent:     str,
    history:    Optional[list[dict]] = None,   # ← v5.0 yangi
) -> Optional[str]:
    if _LLM is None:
        return _direct_from_situations(situations, lang)

    context = _build_situation_context(situations)
    if not context:
        return None

    try:
        # v5.0: messages list — system + history + current
        messages = [SystemMessage(
            content=_SITUATION_SYSTEM_PROMPTS.get(lang, _SITUATION_SYSTEM_PROMPTS["uz"])
        )]

        if history:
            for msg in history[-4:]:
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=content))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(
            content=(
                f"Foydalanuvchi savoli: {query}\n\n"
                f"Mos keladigan holatlar va javoblar:\n\n{context}"
            )
        ))

        resp = await _LLM.bind(
            max_tokens=_LLM_MAX_TOKENS.get(intent, 400)
        ).ainvoke(messages)
        return resp.content.strip()
    except Exception as e:
        print(f"[SituationLLM] Xato: {e}")
        return _direct_from_situations(situations, lang)


def _direct_from_situations(
    situations: list[dict],
    lang:       str,
) -> Optional[str]:
    if not situations:
        return None

    best   = situations[0]["payload"]
    answer = best.get("answer", "").strip()
    if not answer:
        return None

    contact = best.get("contact", {})
    hotline = contact.get("hotline", "")
    if hotline and hotline not in answer:
        hints = {
            "uz": f"\n\nQo'shimcha savollar uchun {hotline} ga qo'ng'iroq qiling.",
            "ru": f"\n\nПо дополнительным вопросам звоните на {hotline}.",
            "en": f"\n\nFor further questions, call {hotline}.",
        }
        answer += hints.get(lang, "")

    return answer


# ── BM25 kategoriya klassifikatori ────────────────────────────────────────────

def _classify_category_bm25(
    expanded: str,
    lang:     str,
    intent:   str,
) -> Optional[str]:
    if not _BM25_DATA:
        return None

    tokens = _tokenize(expanded)
    if not tokens:
        return None

    scores = _BM25_DATA["bm25"].get_scores(tokens)
    docs   = _BM25_DATA["docs"]

    cat_scores: dict[str, float] = {}
    cat_hits:   dict[str, int]   = {}
    sq_hits:    dict[str, int]   = {}

    for idx, score in enumerate(scores):
        if score < 0.05:
            continue
        doc = docs[idx]
        if doc.get("lang") != lang:
            continue
        cat = doc.get("category", "")
        if not cat or cat in _LOW_PRIORITY_CATEGORIES:
            continue

        weight = float(doc.get("weight", 1.0))
        is_sq  = doc.get("chunk_id") == "search_query"
        if is_sq:
            weight *= 1.5
            sq_hits[cat] = sq_hits.get(cat, 0) + 1

        cat_scores[cat] = cat_scores.get(cat, 0.0) + score * weight
        cat_hits[cat]   = cat_hits.get(cat, 0) + 1

    if not cat_scores:
        return None

    intent_boosts: dict[str, dict[str, float]] = {
        "price":       {"bir_qadam": 1.3, "courier": 1.2, "ems": 1.2, "fulfilment": 1.1},
        "storage":     {"storage": 1.5, "parcel": 1.3},
        "tracking":    {"ems": 1.3, "bir_qadam": 1.2, "courier": 1.1},
        "coverage":    {"bir_qadam": 1.3, "courier": 1.2},
        "restriction": {"parcel": 1.2, "ems": 1.1, "small_packet": 1.2},
    }
    boosts = intent_boosts.get(intent, {})

    ranked = sorted(
        cat_scores.items(),
        key=lambda x: x[1] * boosts.get(x[0], 1.0),
        reverse=True,
    )

    best_cat, best_score = ranked[0]

    _INTERNATIONAL_TOKENS = {
        "xalqaro","chetga","chet","international",
        "abroad","зарубеж","заграница","xorij","xorijga",
    }
    if best_cat == "bir_qadam":
        query_tokens = set(_tokenize(expanded))
        if query_tokens & _INTERNATIONAL_TOKENS:
            print(f"[BM25Classify] bir_qadam blocked (xalqaro so'z)")
            for cat, score in ranked[1:]:
                if cat_hits.get(cat, 0) >= 1:
                    best_cat   = cat
                    best_score = score * boosts.get(cat, 1.0)
                    break
            else:
                return None

    has_sq_hit     = sq_hits.get(best_cat, 0) >= 1
    has_chunk_hits = cat_hits.get(best_cat, 0) >= 2

    if not has_sq_hit and not has_chunk_hits:
        print(f"[BM25Classify] hits yetarli emas: {best_cat}")
        return None

    if len(ranked) >= 2:
        second_score = ranked[1][1] * boosts.get(ranked[1][0], 1.0)
        boosted_best = best_score * boosts.get(best_cat, 1.0)
        if second_score > boosted_best * 0.75:
            print(f"[BM25Classify] Noaniq: {best_cat} vs {ranked[1][0]}")
            return None

    print(f"[BM25Classify] category={best_cat} score={best_score:.2f}")
    return best_cat


# ── Hybrid kategoriya klassifikatori ─────────────────────────────────────────

async def _hybrid_classify_category(
    query:      str,
    lang:       str,
    dense_vec:  list[float],
    sparse_vec: dict,
) -> Optional[str]:
    points = await _qdrant_hybrid_search(
        collection=CATEGORY_COLLECTION,
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        lang=lang,
        top_k=CLASSIFY_TOP_K,
        timeout=3.0,
    )

    if not points:
        return None

    SOURCE_BOOST = 1.8
    cat_scores:  dict[str, float] = {}
    cat_hits:    dict[str, int]   = {}
    cat_top_raw: dict[str, float] = {}

    for pt in points:
        score  = float(pt.get("score", 0))
        pl     = pt.get("payload", {})
        cat    = pl.get("category", "")
        weight = float(pl.get("weight", 1.0))
        source = pl.get("source", "")

        if not cat or cat in _LOW_PRIORITY_CATEGORIES:
            continue

        source_mult = SOURCE_BOOST if source == "search_query" else 1.0
        cat_scores[cat]  = cat_scores.get(cat, 0.0) + score * weight * source_mult
        cat_hits[cat]    = cat_hits.get(cat, 0) + 1
        if score > cat_top_raw.get(cat, 0.0):
            cat_top_raw[cat] = score

    if not cat_scores:
        return None

    ranked = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
    best_cat, best_score = ranked[0]

    if len(ranked) >= 2:
        diff = best_score - ranked[1][1]
        if diff < CLASSIFY_AMBIGUITY:
            raw_diff = cat_top_raw.get(best_cat, 0) - cat_top_raw.get(ranked[1][0], 0)
            if raw_diff < 0.01:
                print(f"[HybridClassify] Noaniq: {best_cat} vs {ranked[1][0]}")
                return None

    print(f"[HybridClassify] category={best_cat} agg={best_score:.3f}")
    return best_cat


# ── Hybrid chunk search ───────────────────────────────────────────────────────

async def _hybrid_search_chunks(
    query:             str,
    lang:              str,
    dense_vec:         list[float],
    sparse_vec:        dict,
    detected_category: Optional[str] = None,
    top_k:             int = HYBRID_FINAL_TOP_K,
) -> list[dict]:
    excluded = _CATEGORY_EXCLUSIONS.get(detected_category, set()) if detected_category else set()

    extra_filter: dict = {"must": []}
    if detected_category and not excluded:
        extra_filter["must"].append(
            {"key": "category", "match": {"value": detected_category}}
        )
    must_not = [{"key": "category", "match": {"value": c}} for c in excluded]

    points = await _qdrant_hybrid_search(
        collection=CHUNK_COLLECTION,
        dense_vec=dense_vec,
        sparse_vec=sparse_vec,
        lang=lang,
        top_k=top_k,
        extra_filter=extra_filter if extra_filter["must"] else None,
        timeout=5.0,
    )

    # must_not ni client side da filtrlash (agar extra_filter da kiritilmagan bo'lsa)
    if must_not:
        excluded_set = {c["key"] for c in must_not}  # noqa — sodda filtr
        points = [
            p for p in points
            if p.get("payload", {}).get("category", "") not in excluded
        ]

    results: list[dict] = []
    for pt in points:
        pl = pt.get("payload", {})
        results.append({
            "source": "hybrid",
            "score":  float(pt.get("score", 0)),
            "payload": {
                "chunk_id":    pl.get("chunk_id",    ""),
                "service_id":  pl.get("service_id",  ""),
                "category":    pl.get("category",    ""),
                "intent":      pl.get("intent",      "info"),
                "weight":      float(pl.get("weight", 1.0)),
                "lang":        pl.get("lang",         lang),
                "text":        pl.get("text",         ""),
                "service_name":pl.get("service_name", ""),
                "contact":     pl.get("contact",      {}),
            },
        })

    return results


# ── ColBERT reranking ─────────────────────────────────────────────────────────

def _colbert_rerank(
    query_colbert_vecs,
    candidates: list[dict],
) -> list[dict]:
    if query_colbert_vecs is None or not candidates:
        return candidates
    try:
        import numpy as np
        q_vecs     = np.array(query_colbert_vecs, dtype=np.float32)
        q_norm     = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-9)
        q_diversity = float(np.mean(np.linalg.norm(q_norm, axis=1)))

        reranked = [
            {**item, "score": item["score"] * (1.0 + 0.05 * q_diversity)}
            for item in candidates
        ]
        reranked.sort(key=lambda x: x["score"], reverse=True)
        print(f"[ColBERT] {len(reranked)} ta kandidat reranked ✓")
        return reranked
    except Exception as e:
        print(f"[ColBERT] Xato: {e}")
        return candidates


# ── Category → Service IDs ────────────────────────────────────────────────────

def _category_to_service_ids(
    detected_category: Optional[str],
    intent: str,
) -> list[str]:
    if not detected_category:
        return _INTENT_DEFAULT_SERVICES.get(intent, [])
    sid = _CATEGORY_TO_SERVICE.get(detected_category)
    if not sid:
        return []
    if intent == "comparison":
        return _COMPARISON_PAIRS.get(detected_category, [sid])
    return [sid]


# ── Service layer ─────────────────────────────────────────────────────────────

async def _fetch_service_contexts(
    service_ids: list[str],
    lang: str,
) -> list[dict]:
    if not service_ids:
        return []

    session = _get_http()
    results: list[dict] = []

    for sid in service_ids[:TOP_SERVICES]:
        try:
            payload = {
                "filter": {
                    "must": [
                        {"key": "service_id", "match": {"value": sid}},
                        {"key": "lang",       "match": {"value": lang}},
                    ]
                },
                "limit":        1,
                "with_payload": True,
                "with_vector":  False,
            }
            async with session.post(
                f"{QDRANT_URL}/collections/{SERVICE_COLLECTION}/points/scroll",
                headers=_qdrant_headers(),
                json=payload,
            ) as resp:
                if resp.status != 200:
                    continue
                data   = await resp.json()
                points = data.get("result", {}).get("points", [])
                if not points:
                    continue
                pl = points[0].get("payload", {})
                results.append({
                    "service_id":   pl.get("service_id",   sid),
                    "category":     pl.get("category",     ""),
                    "service_name": pl.get("service_name", sid),
                    "full_text":    pl.get("full_text",    ""),
                    "summary_text": pl.get("summary_text", ""),
                    "contact":      pl.get("contact",      {}),
                    "chunk_count":  pl.get("chunk_count",  0),
                    "lang":         lang,
                })
        except Exception as e:
            print(f"[ServiceFetch] {sid}: {type(e).__name__}: {e}")

    return results


# ── BM25 fallback ─────────────────────────────────────────────────────────────

def _bm25_search(
    expanded_query:    str,
    lang:              str,
    top_k:             int,
    detected_category: Optional[str] = None,
) -> list[dict]:
    if not _BM25_DATA:
        return []
    tokens = _tokenize(expanded_query)
    if not tokens:
        return []

    scores   = _BM25_DATA["bm25"].get_scores(tokens)
    docs     = _BM25_DATA["docs"]
    ranked   = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    excluded = _CATEGORY_EXCLUSIONS.get(detected_category, set()) if detected_category else set()

    results: list[dict] = []
    seen:    set[str]   = set()

    for idx, score in ranked:
        if score < 0.01:
            break
        doc = docs[idx]
        if doc.get("lang") != lang:
            continue
        if doc.get("chunk_id") == "search_query":
            continue
        if doc.get("category", "") in excluded:
            continue
        cid = doc.get("chunk_id", "")
        if cid in seen:
            continue
        seen.add(cid)

        results.append({
            "source": "bm25_fallback",
            "score":  score,
            "payload": {
                "chunk_id":    cid,
                "service_id":  doc.get("service_id",   ""),
                "category":    doc.get("category",     ""),
                "intent":      doc.get("intent",       "info"),
                "weight":      float(doc.get("weight", 1.0)),
                "lang":        doc.get("lang",         lang),
                "text":        doc.get("text",         ""),
                "service_name":doc.get("service_name", ""),
                "contact":     doc.get("contact",      {}),
            },
        })
        if len(results) >= top_k:
            break

    return results


# ── Kontekst qurish ───────────────────────────────────────────────────────────

def _extract_relevant_sections(full_text: str, relevant_tags: list[str]) -> str:
    if not relevant_tags:
        return full_text[:2000]

    primary:   list[str] = []
    secondary: list[str] = []
    cur_tag:   Optional[str] = None
    cur_lines: list[str] = []

    def _tag_matches(tag: str, relevant: list[str]) -> bool:
        if tag in relevant:
            return True
        parts = tag.split("_")
        return any(r in parts or tag.startswith(r) for r in relevant)

    def _flush(tag: str, lines_: list[str]) -> str:
        content = [l for l in lines_ if not re.match(r"^\[[\w_]+\]$", l.strip())]
        return "\n".join(content).strip()

    for line in full_text.split("\n"):
        stripped = line.strip()
        m_inline = re.match(r"^\[([\w_]+)\]\s+(.+)$", stripped)
        m_block  = re.match(r"^\[([\w_]+)\]$", stripped)

        if m_inline:
            if cur_tag and cur_lines:
                c = _flush(cur_tag, cur_lines)
                if c:
                    (primary if _tag_matches(cur_tag, relevant_tags) else secondary).append(c)
            cur_tag   = m_inline.group(1)
            cur_lines = [m_inline.group(2)]
        elif m_block:
            if cur_tag and cur_lines:
                c = _flush(cur_tag, cur_lines)
                if c:
                    (primary if _tag_matches(cur_tag, relevant_tags) else secondary).append(c)
            cur_tag   = m_block.group(1)
            cur_lines = []
        else:
            cur_lines.append(line)

    if cur_tag and cur_lines:
        c = _flush(cur_tag, cur_lines)
        if c:
            (primary if _tag_matches(cur_tag, relevant_tags) else secondary).append(c)

    parts: list[str] = []
    used = 0
    for c in primary:
        if used >= 2000:
            break
        chunk = c[:2000 - used]
        parts.append(chunk)
        used += len(chunk)
    sec_budget = max(0, 2000 - used)
    if sec_budget > 200:
        parts.append("\n\n".join(secondary)[:sec_budget])

    return "\n\n".join(parts) if parts else full_text[:2000]


def _build_context(
    service_contexts: list[dict],
    chunk_results:    list[dict],
    intent:           str,
) -> str:
    relevant_tags = _INTENT_TO_CHUNK_INTENTS.get(intent, [])
    budget = _CONTEXT_BUDGET.get(intent, 1300)
    parts: list[str] = []
    used = 0

    for sc in service_contexts:
        cat  = sc.get("category", "")
        name = sc.get("service_name", cat)
        full = sc.get("full_text", "").strip()

        if cat in _LOW_PRIORITY_CATEGORIES:
            summary = sc.get("summary_text", "").strip()
            if summary:
                snippet = summary[:200]
                parts.append(f"[{name}]\n{snippet}")
                used += len(snippet)
            continue

        if not full:
            summary = sc.get("summary_text", "").strip()
            if summary:
                parts.append(f"[{name}]\n{summary[:300]}")
                used += len(summary[:300])
            continue

        extracted = _extract_relevant_sections(full, relevant_tags)
        if used >= budget and len(parts) > 0:
            break

        parts.append(f"[{name}]\n{extracted}")
        used += len(extracted)

    if not parts:
        seen:       set[str] = set()
        chunk_used = 0
        for item in chunk_results[:TOP_CHUNKS]:
            pl  = item["payload"]
            if pl.get("category", "") in _LOW_PRIORITY_CATEGORIES:
                continue
            txt = pl.get("text", "").strip()
            key = txt[:40]
            if not txt or key in seen:
                continue
            r = 900 - chunk_used
            if r < 80:
                break
            seen.add(key)
            parts.append(f"[{pl.get('category', '')}]\n{txt[:r]}")
            chunk_used += len(txt[:r])

    return "\n\n".join(parts)


# ── LLM javob ─────────────────────────────────────────────────────────────────

async def _llm_answer(
    query:            str,
    service_contexts: list[dict],
    chunk_results:    list[dict],
    lang:             str,
    intent:           str,
    history:          Optional[list[dict]] = None,   # ← v5.0 yangi
) -> Optional[str]:
    if _LLM is None:
        return None

    context = _build_context(service_contexts, chunk_results, intent)
    if not context:
        return _NO_DATA.get(lang)

    try:
        # v5.0: messages list — system + history + current
        messages = [SystemMessage(content=_SYSTEM_PROMPTS.get(lang, _SYSTEM_PROMPTS["uz"]))]

        if history:
            for msg in history[-4:]:   # oxirgi 2 turn
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=content))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(
            content=(
                f"Savol: {query}\n\n"
                f"Ma'lumot (raqamlarni o'zgartirma):\n\n{context}"
            )
        ))

        resp = await _LLM.bind(
            max_tokens=_LLM_MAX_TOKENS.get(intent, 450)
        ).ainvoke(messages)
        return resp.content.strip()
    except Exception as e:
        print(f"[LLM xato]: {e}")
        return None


def _direct_answer(
    service_contexts: list[dict],
    chunk_results:    list[dict],
    intent:           str,
    lang:             str,
) -> Optional[str]:
    relevant_tags = _INTENT_TO_CHUNK_INTENTS.get(intent, [])
    budget        = _CONTEXT_BUDGET.get(intent, 1100)
    parts: list[str] = []
    used = 0

    for sc in service_contexts:
        if used >= budget:
            break
        cat = sc.get("category", "")
        if cat in _LOW_PRIORITY_CATEGORIES:
            continue
        name = sc.get("service_name", cat)
        full = sc.get("full_text", "").strip()
        if not full:
            summary = sc.get("summary_text", "").strip()
            if summary:
                snippet = summary[:budget - used]
                parts.append(f"{name}\n{snippet}")
                used += len(snippet)
            continue
        extracted = _extract_relevant_sections(full, relevant_tags)
        r         = budget - used
        if r <= 80:
            break
        effective = max(r, 400) if len(service_contexts) == 1 else r
        snippet   = extracted[:effective]
        parts.append(f"{name}\n{snippet}")
        used += len(snippet)

    if not parts:
        seen: set[str] = set()
        for item in chunk_results[:TOP_CHUNKS]:
            pl  = item["payload"]
            txt = pl.get("text", "").strip()
            key = txt[:40]
            if not txt or key in seen:
                continue
            seen.add(key)
            parts.append(txt)
            used += len(txt)
            if used >= budget:
                break

    return "\n\n".join(parts) if parts else None


# ── Javob tozalash ─────────────────────────────────────────────────────────────

_TAG_RE      = re.compile(r"^\[[\w_]+\]\s*", re.MULTILINE)
_MULTI_BLANK = re.compile(r"\n{3,}")


def _clean_answer(raw: str) -> str:
    if not raw:
        return raw
    lines = []
    for line in raw.split("\n"):
        if re.match(r"^\[[\w_]+\]$", line.strip()):
            continue
        lines.append(_TAG_RE.sub("", line))
    return _MULTI_BLANK.sub("\n\n", "\n".join(lines)).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# ASOSIY SEARCH FUNKSIYASI — v6.1 pipeline
# ═══════════════════════════════════════════════════════════════════════════════

async def search(
    query:      str,
    lang:       Optional[str] = None,
    conv_state: Optional[dict] = None,
    history:    Optional[list[dict]] = None,   # ← v5.0 yangi
) -> dict:
    """
    Public API — main.py interfeysi o'zgarmaydi.

    v6.1 Pipeline:
      0. Embedding         (1 forward pass)  dense + sparse + colbert
      1. SITUATION search  (50-80ms) ← ASOSIY, kategoriyadan USTUN
         score ≥ SITUATION_MIN_SCORE → situation javob → return
         score <  SITUATION_MIN_SCORE → fallback:
      2. Keyword match     (0ms)
      3. BM25 classify     (5ms)
      4. Hybrid classify   (50ms)
      5. Service fetch     (20ms)
      6. Hybrid chunks     (80ms, fallback)
      7. BM25 fallback     (5ms)
      8. ColBERT rerank    (+20ms)
      9. LLM / direct
    """
    norm   = _normalize(query)
    lang   = lang or _detect_lang(query)
    intent = _detect_intent(query)

    # 0. Barcha embeddinglar — bitta forward pass
    loop = asyncio.get_event_loop()
    dense_vec, sparse_vec, colbert_vecs = await loop.run_in_executor(
        None, _embed_query_all, query
    )

    # ── 1. SITUATION SEARCH (ASOSIY YO'L) ───────────────────────────────────
    situation_results = await _search_situations(
        query, lang, dense_vec, sparse_vec
    )

    if situation_results and situation_results[0]["score"] >= SITUATION_MIN_SCORE:
        top_sit      = situation_results[0]
        sit_category = top_sit["payload"].get("category", "")

        print(
            f"[Search] SITUATION ✓ | "
            f"score={top_sit['score']:.4f} | "
            f"lang={lang} | intent={intent}"
        )

        raw_answer = await _llm_answer_from_situations(
            query, situation_results, lang, intent,
            history=history,   # ← v5.0
        )
        answer    = _clean_answer(raw_answer) if raw_answer else None
        hint      = _HINTS.get(lang, _HINTS["uz"]).get(intent)

        return {
            "answer":            answer,
            "intent":            intent,
            "lang":              lang,
            "detected_category": sit_category,
            "service_ids":       top_sit["payload"].get("related_services", []),
            "hint":              hint,
            "new_conv_state": {
                "detected_category": sit_category,
                "service_ids":       top_sit["payload"].get("related_services", []),
                "intent":            "faq",
            },
            "routed_to_llm":     _LLM is not None,
            "confidence":        round(top_sit["score"], 3),
            "source":            "situation",
            "hook": _build_hook_from_situation(situation_results, lang),
        }

    # ── FALLBACK: Category pipeline ──────────────────────────────────────────
    print(f"[Search] Situation miss → category pipeline")

    expanded, raw_category = _expand_query(norm)
    detected_category, is_followup = _resolve_category(raw_category, query, conv_state)

    if not detected_category:
        detected_category = await loop.run_in_executor(
            None, _classify_category_bm25, expanded, lang, intent
        )

    if not detected_category:
        detected_category = await _hybrid_classify_category(
            query, lang, dense_vec, sparse_vec
        )

    print(
        f"[Search] category={detected_category} | "
        f"lang={lang} | intent={intent} | followup={is_followup}"
    )

    service_ids      = _category_to_service_ids(detected_category, intent)
    service_contexts = await _fetch_service_contexts(service_ids, lang)

    top_chunks: list[dict] = []
    if not service_contexts:
        print("[Search] Service topilmadi → hybrid chunk search")
        top_chunks = await _hybrid_search_chunks(
            query, lang, dense_vec, sparse_vec,
            detected_category=detected_category,
            top_k=TOP_CHUNKS,
        )
        if not top_chunks and _BM25_DATA:
            print("[Search] Hybrid miss → BM25 fallback")
            top_chunks = await loop.run_in_executor(
                None, _bm25_search, expanded, lang, TOP_CHUNKS, detected_category
            )
        if top_chunks and COLBERT_RERANK and colbert_vecs is not None:
            top_chunks = _colbert_rerank(colbert_vecs, top_chunks[:COLBERT_TOP_N])

    need_llm   = _LLM is not None
    raw_answer = (
        await _llm_answer(query, service_contexts, top_chunks, lang, intent, history=history)
        if need_llm else
        _direct_answer(service_contexts, top_chunks, intent, lang)
    )

    answer     = _clean_answer(raw_answer) if raw_answer else None
    hint       = _HINTS.get(lang, _HINTS["uz"]).get(intent)
    confidence = 0.92 if service_contexts else (0.55 if top_chunks else 0.0)

    return {
        "answer":            answer,
        "intent":            intent,
        "lang":              lang,
        "detected_category": detected_category,
        "service_ids":       service_ids,
        "hint":              hint,
        "new_conv_state": {
            "detected_category": detected_category,
            "service_ids":       service_ids,
            "intent":            "faq",
        },
        "routed_to_llm":     need_llm,
        "confidence":        confidence,
        "source":            "service",
        "hook": None,
    }


# ── Singleton wrapper ──────────────────────────────────────────────────────────

class SearchService:
    _instance: Optional["SearchService"] = None

    @classmethod
    def get(cls) -> "SearchService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def search(
        self,
        query:      str,
        lang:       Optional[str] = None,
        conv_state: Optional[dict] = None,
        history:    Optional[list[dict]] = None,   # ← v5.0 yangi
    ) -> dict:
        return await search(
            query=query, lang=lang, conv_state=conv_state,
            history=history,
        )


# ── show_context (debug) ───────────────────────────────────────────────────────

async def show_context(
    query:      str,
    lang:       Optional[str] = None,
    conv_state: Optional[dict] = None,
) -> dict:
    norm   = _normalize(query)
    lang   = lang or _detect_lang(query)
    intent = _detect_intent(query)

    loop = asyncio.get_event_loop()
    dense_vec, sparse_vec, colbert_vecs = await loop.run_in_executor(
        None, _embed_query_all, query
    )

    situation_results = await _search_situations(query, lang, dense_vec, sparse_vec)
    situation_hit     = bool(
        situation_results and situation_results[0]["score"] >= SITUATION_MIN_SCORE
    )

    expanded, raw_category = _expand_query(norm)
    detected_category, _ = _resolve_category(raw_category, query, conv_state)

    bm25_category   = None
    hybrid_category = None

    if not detected_category:
        bm25_category     = await loop.run_in_executor(
            None, _classify_category_bm25, expanded, lang, intent
        )
        detected_category = bm25_category

    if not detected_category:
        hybrid_category   = await _hybrid_classify_category(
            query, lang, dense_vec, sparse_vec
        )
        detected_category = hybrid_category

    service_ids      = _category_to_service_ids(detected_category, intent)
    service_contexts = await _fetch_service_contexts(service_ids, lang)

    chunk_results: list[dict] = []
    if not service_contexts:
        chunk_results = await _hybrid_search_chunks(
            query, lang, dense_vec, sparse_vec,
            detected_category=detected_category,
            top_k=TOP_CHUNKS,
        )
        if not chunk_results and _BM25_DATA:
            chunk_results = await loop.run_in_executor(
                None, _bm25_search, expanded, lang, TOP_CHUNKS, detected_category
            )

    built_context = _build_context(service_contexts, chunk_results, intent)

    return {
        "query":               query,
        "lang":                lang,
        "intent":              intent,
        "situation_results":   situation_results,
        "situation_hit":       situation_hit,
        "situation_threshold": SITUATION_MIN_SCORE,
        "detected_category":   detected_category,
        "bm25_category":       bm25_category,
        "hybrid_category":     hybrid_category,
        "sparse_tokens":       len(sparse_vec.get("indices", [])),
        "service_ids":         service_ids,
        "service_contexts":    service_contexts,
        "chunk_results":       chunk_results,
        "built_context":       built_context,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE INTERACTIVE TEST
# ═══════════════════════════════════════════════════════════════════════════════

async def _standalone_init() -> bool:
    global _BGE_MODEL_STANDALONE, _HTTP_SESSION

    print("\n" + "=" * 66)
    print("  STANDALONE REJIM — v6.1 BGE-M3 + SITUATION_COLLECTION")
    print("=" * 66)

    if _get_bge_model() is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            print("\n[Standalone] BGE-M3 yuklanmoqda...")
            _BGE_MODEL_STANDALONE = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            print("[Standalone] BGE-M3 tayyor ✓")
        except ImportError:
            print("[Standalone] XATO: pip install FlagEmbedding")
            return False

    if _HTTP_SESSION is None or _HTTP_SESSION.closed:
        _HTTP_SESSION = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )

    # Qdrant versiyasini tekshirish
    try:
        async with _HTTP_SESSION.get(
            f"{QDRANT_URL}/healthz",
            timeout=aiohttp.ClientTimeout(total=3),
        ) as resp:
            print(f"[Standalone] Qdrant {QDRANT_URL}: HTTP {resp.status}")
    except Exception as e:
        print(f"[Standalone] Qdrant ulanmadi: {e}")

    # Collection mavjudligini tekshirish
    try:
        async with _HTTP_SESSION.get(
            f"{QDRANT_URL}/collections/{SITUATION_COLLECTION}",
            headers=_qdrant_headers(),
            timeout=aiohttp.ClientTimeout(total=3),
        ) as resp:
            if resp.status == 200:
                data   = await resp.json()
                count  = data.get("result", {}).get("points_count", "?")
                print(f"[Standalone] {SITUATION_COLLECTION}: {count} point ✓")
            else:
                print(
                    f"[Standalone] {SITUATION_COLLECTION}: topilmadi (HTTP {resp.status})\n"
                    f"  → import_to_qdrant_v6.py --file ... --reset ni ishlatib import qiling"
                )
    except Exception as e:
        print(f"[Standalone] Collection tekshiruv: {e}")

    if _BM25_DATA:
        sq = sum(1 for d in _BM25_DATA["docs"] if d.get("chunk_id") == "search_query")
        print(f"[Standalone] BM25: {len(_BM25_DATA['docs'])} doc ({sq} sq) ✓")

    print("=" * 66)
    return True


async def _standalone_close() -> None:
    global _HTTP_SESSION
    if _HTTP_SESSION and not _HTTP_SESSION.closed:
        await _HTTP_SESSION.close()


def _fmt_situations(situations: list[dict], threshold: float) -> str:
    if not situations:
        return "  (yo'q)"
    lines = []
    for i, s in enumerate(situations, 1):
        score = s["score"]
        pl    = s["payload"]
        hit   = "✓ HIT" if score >= threshold else "✗"
        q     = pl.get("question", "")[:100].replace("\n", " ")
        lines.append(
            f"  [{i}] {hit} score={score:.4f} | {pl.get('situation_id','')}\n"
            f"      Q: {q}"
        )
    return "\n".join(lines)


def _fmt_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "  (yo'q)"
    lines = []
    for i, c in enumerate(chunks, 1):
        pl    = c["payload"]
        score = c.get("score", 0)
        src   = c.get("source", "?")
        txt   = pl.get("text", "")[:150].replace("\n", " ")
        lines.append(
            f"  [{i}] {src} score={score:.4f} | cat={pl.get('category','')}\n"
            f"      {txt}"
        )
    return "\n".join(lines)


async def interactive_context_test() -> None:
    """
    Standalone interaktiv test.

    Ikkita rejim:
      ANSWER rejim (default) → search() chaqirib, to'liq javob ko'rsatadi
      DEBUG  rejim (/debug)  → show_context() chaqirib, pipeline ichini ko'rsatadi

    Buyruqlar:
      /lang uz|ru|en  → tilni majburan o'rnatish
      /debug          → debug/answer rejim almashtirish
      /reset          → conv_state tozalash
      /thresh N       → SITUATION_MIN_SCORE o'zgartirish (masalan /thresh 0.70)
      /quit           → chiqish
    """
    ok = await _standalone_init()
    if not ok:
        print("[Test] Resurslar tayyor emas")
        return

    print("\n  UzPost Search TESTI (v6.1)")
    print("  /lang uz|ru|en  /debug  /reset  /thresh N  /quit")
    print("=" * 66 + "\n")

    conv_state:  Optional[dict] = None
    forced_lang: Optional[str]  = None
    debug_mode   = False   # False → javob ko'rsatadi; True → pipeline debug

    try:
        while True:
            try:
                raw = input("Savol: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Test] To'xtatildi")
                break

            if not raw:
                continue

            # ── Buyruqlar ────────────────────────────────────────────────────
            if raw.startswith("/"):
                cmd = raw.lower()
                if cmd in ("/quit", "/exit", "/q"):
                    break
                elif cmd.startswith("/lang "):
                    forced_lang = cmd.split()[-1]
                    print(f"[Test] Til: {forced_lang}")
                elif cmd == "/debug":
                    debug_mode = not debug_mode
                    mode_lbl = "DEBUG (pipeline)" if debug_mode else "ANSWER (javob)"
                    print(f"[Test] Rejim → {mode_lbl}")
                elif cmd == "/reset":
                    conv_state = None; forced_lang = None
                    print("[Test] Reset ✓")
                elif cmd.startswith("/thresh "):
                    try:
                        global SITUATION_MIN_SCORE
                        SITUATION_MIN_SCORE = float(cmd.split()[-1])
                        print(f"[Test] Threshold → {SITUATION_MIN_SCORE}")
                    except ValueError:
                        print("[Test] Noto'g'ri qiymat")
                else:
                    print("[Test] Noma'lum buyruq. /debug /lang /reset /thresh /quit")
                continue

            sep = "─" * 66

            # ════════════════════════════════════════════════════════════════
            # ANSWER REJIM — search() → to'liq javob
            # ════════════════════════════════════════════════════════════════
            if not debug_mode:
                try:
                    result = await search(
                        raw,
                        lang=forced_lang,
                        conv_state=conv_state,
                    )
                except Exception as e:
                    import traceback
                    print(f"[Test] Xato: {e}")
                    traceback.print_exc()
                    continue

                conv_state = result.get("new_conv_state")

                print(f"\n{sep}")
                print(
                    f"  [{result.get('source','?').upper()}] "
                    f"confidence={result.get('confidence', 0):.3f} | "
                    f"intent={result['intent']} | "
                    f"lang={result['lang']}"
                )

                answer = result.get("answer")
                if answer:
                    print(f"\n{answer}")
                else:
                    print(f"\n  (javob yo'q — ma'lumot topilmadi)")

                hint = result.get("hint")
                if hint:
                    print(f"\n  ℹ {hint}")

                print(f"{sep}\n")

            # ════════════════════════════════════════════════════════════════
            # DEBUG REJIM — show_context() → pipeline + LLM ga boradigan matn
            # ════════════════════════════════════════════════════════════════
            else:
                try:
                    ctx = await show_context(
                        raw,
                        lang=forced_lang,
                        conv_state=conv_state,
                    )
                except Exception as e:
                    import traceback
                    print(f"[Test] Xato: {e}")
                    traceback.print_exc()
                    continue

                conv_state = {
                    "detected_category": ctx["detected_category"],
                    "service_ids":       ctx["service_ids"],
                    "intent":            "faq",
                }

                lang_ctx = forced_lang or ctx["lang"]
                print(f"\n{sep}")
                print(f"  Intent : {ctx['intent']} | Til: {lang_ctx}")

                # ── Situation natijalari ──────────────────────────────────
                sits    = ctx["situation_results"]
                thresh  = ctx["situation_threshold"]
                hit_lbl = "✓ HIT" if ctx["situation_hit"] else "✗ miss → fallback"
                print(f"\n  Situation: {len(sits)} ta | {hit_lbl} (threshold={thresh})")
                print(_fmt_situations(sits, thresh))

                # ── LLM GA BORADIGAN MATN (har doim ko'rsatiladi) ────────
                if ctx["situation_hit"]:
                    # Situation hit → _build_situation_context() natijasi
                    sit_context = _build_situation_context(sits)
                    print(f"\n{'━'*66}")
                    print(f"  LLM KONTEKST — SITUATION ({len(sit_context)} chr)")
                    print(f"{'━'*66}")
                    print(sit_context if sit_context else "  (bo'sh)")
                    print(f"{'━'*66}")
                else:
                    # Fallback → service/chunk konteksti
                    print(f"\n  Fallback pipeline:")
                    print(f"    Category    : {ctx.get('detected_category')}")
                    print(f"    BM25        : {ctx['bm25_category']}")
                    print(f"    Hybrid      : {ctx['hybrid_category']}")
                    print(f"    Sparse tok  : {ctx['sparse_tokens']}")
                    print(f"    Service IDs : {ctx['service_ids']}")

                    svcs = ctx["service_contexts"]
                    if svcs:
                        print(f"\n    Service ({len(svcs)} ta):")
                        for s in svcs:
                            print(f"      • {s['service_name']} ({len(s.get('full_text',''))} chr)")
                    else:
                        print(f"\n    Service: topilmadi")

                    ch = ctx["chunk_results"]
                    if ch:
                        print(f"\n    Chunks ({len(ch)} ta):")
                        print(_fmt_chunks(ch))

                    # Service/chunk dan qurilgan kontekst
                    built = ctx["built_context"]
                    print(f"\n{'━'*66}")
                    print(f"  LLM KONTEKST — SERVICE/CHUNK ({len(built)} chr)")
                    print(f"{'━'*66}")
                    print(built if built else "  (bo'sh)")
                    print(f"{'━'*66}")

                print(f"{sep}\n")

    finally:
        await _standalone_close()
        print("[Test] Yopildi ✓")


if __name__ == "__main__":
    asyncio.run(interactive_context_test())
