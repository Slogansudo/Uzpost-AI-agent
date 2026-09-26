
import os
import re
import hashlib
import time
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from shared_resources import init_all as _init_shared, close_all as _close_shared
from faq.search_service import SearchService, init_resources, close_resources
from intent_classifier import classify_intent
from db_models.session_services import (
    init_db_pool, add_message, get_history,
    get_global_cache, set_global_cache,
)
from admin_api import router as admin_router

# ── YANGI: ConvStateManager import ──────────────────────────────────────────
from conv_state_manager import ConvStateManager, ResumeResult

# ── YANGI: Conversation context tizimi ──────────────────────────────────────
from conversation import strip_for_context
# ── YANGI: Entity memory + follow-up (suhbat boshqaruvi qatlami) ─────────────
from conversation.entities import (
    remember_entities,
    get_last_intent,
    get_remembered_barcode,
    barcode_from_history,
    looks_like_new_topic,
    is_clear_decline,
    is_clear_accept,
    is_identity,
    clear_entities,
    set_offer,
    get_offer,
    clear_offer,
)

load_dotenv()

app = FastAPI(title="UzPost AI", root_path="/aiasistant")
app.include_router(admin_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ─── IN-MEMORY ─────────────────────────────────────────────────────────────────

_CONV_STATE:   dict[str, dict]       = {}
_CHAT_HISTORY: dict[str, list[dict]] = {}
HISTORY_MAX = 5
GROQ_MODEL = os.getenv("GROQ_MODEL_PREMIUM")
_QUERY_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL = {
    "location": 60 * 60 * 24,
    "faq":      60 * 60 * 6,
}

_STATE_MGR = ConvStateManager()


def _cache_get(intent: str, query: str, lang: str) -> Optional[str]:
    key   = f"{intent}:{lang}:{hashlib.md5(query.lower().strip().encode()).hexdigest()}"
    entry = _QUERY_CACHE.get(key)
    if not entry:
        return None
    answer, ts = entry
    if time.time() - ts > _CACHE_TTL.get(intent, 300):
        del _QUERY_CACHE[key]
        return None
    return answer


def _cache_set(intent: str, query: str, lang: str, answer: str) -> None:
    key = f"{intent}:{lang}:{hashlib.md5(query.lower().strip().encode()).hexdigest()}"
    _QUERY_CACHE[key] = (answer, time.time())



_FAQ_LLM:      Optional[ChatGroq] = None
_OFFTOPIC_LLM: Optional[ChatGroq] = None


def _init_llm_clients() -> None:
    global _FAQ_LLM, _OFFTOPIC_LLM
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        print("[LLM] GROQ_API_KEY topilmadi")
        return
    _FAQ_LLM = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=480,
        api_key=key,
    )
    _OFFTOPIC_LLM = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.35,
        max_tokens=420,
        api_key=key,
    )
    print("[LLM] FAQ + Offtopic LLM tayyor ✓")



_AGENTS: dict = {}


def _get_agents() -> dict:
    global _AGENTS
    if not _AGENTS:
        from tracking.agent import TrackingAgent
        from calculator.agent import PriceAgent
        _AGENTS = {
            "tracking": TrackingAgent(),
            "price":    PriceAgent(),
        }
    return _AGENTS



@app.on_event("startup")
async def startup():
    await init_db_pool()
    print("[Startup] DB pool tayyor ✓")
    await _init_shared()
    await init_resources()
    _init_llm_clients()
    _get_agents()

    from admin_api import set_pool
    from db_models.session_services import _pool
    set_pool(_pool)
    print("[Startup] Barcha komponentlar tayyor ✓")


@app.on_event("shutdown")
async def shutdown():
    await close_resources()
    await _close_shared()
    print("[Shutdown] Barcha resurslar yopildi ✓")



class QueryRequest(BaseModel):
    query:      str
    lang:       str            = "uz"
    session_id: Optional[str] = None
    browser_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer:     str
    cached:     bool = False
    session_id: str
    intent:     str  = ""
    lang:       str  = "uz"


class HistoryMessage(BaseModel):
    role:    str
    content: str
    ts:      str


class HistoryResponse(BaseModel):
    messages: list[HistoryMessage]
    user_key: str


def _user_key(req: Request, browser_id: Optional[str]) -> str:
    client_ip = req.client.host if req.client else "unknown"
    ua        = req.headers.get("user-agent", "")
    return (
        browser_id
        or req.headers.get("X-User-Key")
        or hashlib.md5(f"{client_ip}-{ua}".encode()).hexdigest()
    )


def _validate_lang(lang: str) -> str:
    return lang if lang in ("uz", "ru", "en") else "uz"


def _get_conv_state(user_key: str) -> Optional[dict]:
    return _CONV_STATE.get(user_key)

def _set_conv_state(user_key: str, state: dict) -> None:
    _CONV_STATE[user_key] = state

def _clear_conv_state(user_key: str) -> None:
    _CONV_STATE.pop(user_key, None)


def _history_push(user_key: str, query: str, answer: str = "") -> None:
    if user_key not in _CHAT_HISTORY:
        _CHAT_HISTORY[user_key] = []
    _CHAT_HISTORY[user_key].append({"role": "user", "content": query})
    if answer:
        # max_len=600: uzunroq javoblarda ham oxiridagi taklif ("...beraymi?")
        # to'liq saqlanadi — "ha" deganda detektor uni o'qiy oladi.
        clean = strip_for_context(answer, max_len=600)
        if clean:
            _CHAT_HISTORY[user_key].append({"role": "assistant", "content": clean})
    # 5 turn = 10 xabar (user + assistant)
    if len(_CHAT_HISTORY[user_key]) > HISTORY_MAX * 2:
        _CHAT_HISTORY[user_key] = _CHAT_HISTORY[user_key][-HISTORY_MAX * 2:]

def _history_get(user_key: str) -> list[dict]:
    return _CHAT_HISTORY.get(user_key, [])


_BARCODE_STRIP_RE = re.compile(r"\b[A-Z]{2,4}\d{7,14}[A-Z]{0,2}\b", re.IGNORECASE)
_LATLON_STRIP_RE = re.compile(r"\(?\s*-?\d{1,3}\.\d{3,8}\s*,\s*-?\d{1,3}\.\d{3,8}\s*\)?")


def _build_context(history: list[dict], current: str) -> str:
    if not history:
        return current
    lines = ["=== OLDINGI SAVOLLAR ==="]
    for msg in history:
        clean = _BARCODE_STRIP_RE.sub("[trek_raqam]", msg["content"])
        clean = _LATLON_STRIP_RE.sub("[koordinata]", clean)
        lines.append(f"- {clean}")
    lines.append("=== JORIY SAVOL ===")
    lines.append(current)
    lines.append("\nQOIDA: Intent FAQAT JORIY SAVOL asosida aniqlansin.")
    return "\n".join(lines)


async def _save(user_key: str, query: str, answer: str, intent: str | None = None) -> None:
    """User+bot xabarini in-memory va DB ga saqlash.
    Agar javob 'qo'shimcha savol' (hook) bilan tugasa — pending offer'ni belgilaymiz
    (history strip qilingani uchun keyingi turnda shu belgi orqali ha/yo'q aniqlanadi)."""
    if intent in ("tracking", "location", "price") and answer:
        low = answer.lower()
        if "💬" in answer or "boshqa yo'nalish" in low or "another route" in low or "другой маршрут" in low:
            set_offer(user_key, intent)
    _history_push(user_key, query, answer)   # ← v5.0: answer qo'shildi
    await add_message(user_key, "user", query)
    await add_message(user_key, "assistant", answer)


def _ends_with_question(text: str) -> bool:

    if not text:
        return False
    if "💬" in text:           # allaqachon bir hook bor
        return True
    # Oxirgi mazmunli qatorda "?" bo'lsa — bu follow-up savol
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line:
            return line.endswith("?") or line.endswith("?")
    return False


def _append_hint(answer: str, resume: ResumeResult) -> str:

    if not resume.resume_hint:
        return answer
    if _ends_with_question(answer):
        return answer
    return f"{answer}\n\n💬 {resume.resume_hint}"



_PRICE_EXIT_RE = re.compile(
    r"\b(bekor|yangi\s+so'rov|boshqa\s+savol|to'xtat|stop|cancel|ortga|back|"
    r"yo'?q|yoq|kerakmas|kerak\s*emas|shart\s*emas|hojati\s*yo'?q|bo'?lmaydi|bas|"
    r"отмена|заново|другой\s+вопрос|стоп|назад|нет|не\s*надо|не\s*нужно|незачем|хватит)\b",
    re.I | re.U,
)

def _is_price_exit(query: str) -> bool:
    return bool(_PRICE_EXIT_RE.search(query))


# ─── PRICE SLOT DETECTOR (interrupt vs davom ettirish) ──────────────────────────
# Xabar price uchun "slot javobi" (og'irlik / shahar / tuman / xizmat raqami)mi?
# Bo'lsa — price davom etadi. Bo'lmasa (haqiqiy FAQ/tracking/location savoli) —
# task interrupt qilinadi (pause + boshqa intentga o'tkaziladi).
_PRICE_SLOT_WEIGHT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(kg|кг|kilo|kilogram)\b", re.I | re.U)
_PRICE_SLOT_SUFFIX_RE = re.compile(r"\b[\w'`-]{3,}(dan|дан|gacha|гача|ga|га)\b", re.I | re.U)


def _looks_like_price_slot(query: str) -> bool:
    q = query.strip()
    words = q.split()
    # sof og'irlik javobi: "2kg", "1.5 kg"
    if len(words) <= 2 and _PRICE_SLOT_WEIGHT_RE.search(q):
        return True
    # qisqa, kelishikli manzil: "Toshkentdan", "Turkiyaga 2kg", "Buxorodan Toshkentga 2kg"
    if len(words) <= 3 and _PRICE_SLOT_SUFFIX_RE.search(q):
        return True
    # qisqa yalang javob (shahar/tuman nomi yoki "1") — domen so'zi bo'lmasa
    if len(words) <= 2 and not looks_like_new_topic(q):
        return True
    return False


# ─── ENGAGEMENT HOOK REPLY (ha/yo'q) ───────────────────────────────────────────
# Agent javobi oxirida "qo'shimcha savol" (hook) beradi. Mijoz "ha" desa — o'sha
# intent BOSHIDAN qayta boshlanadi; "yo'q" desa — muloyim yopiladi. Bu markazlashgan
# logika; agentlarning o'ziga tegmaydi.

# Bot oxirgi xabarida "davom etamizmi?" tipidagi taklif berdimi?
_OFFER_HOOK_RE = re.compile(
    r"💬|boshqa\s+yo'?nalish|boshqa\s+hudud|boshqa\s+jo'?natma|"
    r"другой\s+маршрут|другом\s+район|другое\s+отправлен|"
    r"another\s+route|another\s+area|another\s+shipment",
    re.I | re.U,
)


def _bot_made_offer(bot_msg: str) -> bool:
    return bool(bot_msg and _OFFER_HOOK_RE.search(bot_msg))


def _last_bot_msg(history: list[dict] | None) -> str:
    if not history:
        return ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return (msg.get("content") or "").strip()
    return ""


# Hook'ga "ha" — intentni boshidan boshlash uchun birinchi savol
_RESTART_PROMPT = {
    "price": {
        "uz": "Albatta! Qayerdan, qayerga va taxminan necha kg jo'natmoqchisiz? "
              "Masalan: \"Toshkentdan Buxoroga 2kg\".",
        "ru": "Конечно! Откуда, куда и примерно сколько кг отправляете? "
              "Например: \"из Ташкента в Бухару 2кг\".",
        "en": "Sure! From where, to where, and about how many kg? "
              "For example: \"from Tashkent to Bukhara 2kg\".",
    },
    "location": {
        "uz": "Albatta! Qaysi hudud yoki manzil bo'yicha pochta bo'limini topay? "
              "Manzilni yozing yoki joylashuvingizni yuboring.",
        "ru": "Конечно! По какому району или адресу найти отделение? "
              "Напишите адрес или отправьте геолокацию.",
        "en": "Sure! Which area or address should I search? "
              "Type the address or share your location.",
    },
    "tracking": {
        "uz": "Albatta! Kuzatish uchun trek raqamingizni yuboring.",
        "ru": "Конечно! Пришлите трек-номер для отслеживания.",
        "en": "Sure! Send the tracking number to track it.",
    },
}

# Hook'ga "yo'q" — muloyim yopish (barcha intentlar uchun bir xil)
_DECLINE_MSG = {
    "uz": "Ho'p! Boshqa savolingiz bo'lsa, bemalol yozing — yordam berishga tayyorman.",
    "ru": "Хорошо! Если будет другой вопрос — пишите, всегда готов помочь.",
    "en": "Alright! If you have another question, just write — I'm here to help.",
}

# Location uchun standart (faqat location'ga oid) hook — cross-intent taklifni almashtiradi
_LOC_HOOK_STD = {
    "uz": "Boshqa hudud yoki manzil bo'yicha ham pochta bo'limini topib beraymi?",
    "ru": "Найти отделение по другому району или адресу?",
    "en": "Shall I find a post office for another area or address?",
}


def _force_location_hook(answer: str, lang: str) -> str:
    """Location javobidagi hook'ni faqat-location taklifiga almashtiradi (agar hook bo'lsa)."""
    if "💬" not in answer:
        return answer
    base = answer.split("💬")[0].rstrip()
    return f"{base}\n\n💬 {_LOC_HOOK_STD.get(lang, _LOC_HOOK_STD['uz'])}"


def _close_all(user_key: str) -> None:
    """Barcha holatni to'liq yopadi: conv_state, state manager, price sessiyasi, eslab qolingan mavzu."""
    _clear_conv_state(user_key)
    try:
        _STATE_MGR.close(user_key)
    except Exception:
        pass
    try:
        _get_agents().get("price").clear(user_key)
    except Exception:
        pass
    clear_entities(user_key)


# ─── PRICE MULTI-TURN HANDLER ──────────────────────────────────────────────────

async def _handle_price_multiturn(
    query:      str,
    lang:       str,
    user_key:   str,
    conv_state: dict,
) -> tuple:
    """
    Price multi-turn ni to'liq boshqaradi.
    Qaytaradi: (answer: str | None, should_return: bool, fallback_intent: str | None)
    """
    agent = _get_agents().get("price")

    # ── Bekor qilish ──────────────────────────────────────────────────────────
    if _is_price_exit(query):
        _close_all(user_key)
        exit_msg = {
            "uz": "Narx hisoblash bekor qilindi. Boshqa savolingiz bo'lsa yozing.",
            "ru": "Расчёт отменён. Задайте другой вопрос.",
            "en": "Price calculation cancelled. Feel free to ask something else.",
        }.get(lang, "Bekor qilindi.")
        await _save(user_key, query, exit_msg)
        return exit_msg, True, None

    current_turn = conv_state.get("turn", 1)
    in_service_step = conv_state.get("in_service_step", False)

    print(f"[PriceMultiTurn] turn={current_turn} | service_step={in_service_step}")

    # ── Xizmat tanlash bosqichi ───────────────────────────────────────────────
    if in_service_step:
        agent_response = await agent.run(
            user_key=user_key,
            query=query,
            lang=lang,
            turn=current_turn,
        )
        if agent_response.needs_more and agent_response.new_conv_state:
            _set_conv_state(user_key, agent_response.new_conv_state)
            _STATE_MGR.set_active(user_key, "price", agent_response.new_conv_state)
        else:
            _clear_conv_state(user_key)
            _STATE_MGR.close(user_key)       # hisob tugadi → stale paused price qolmaydi
            agent.clear(user_key)
        await _save(user_key, query, agent_response.answer, intent="price")
        return agent_response.answer, True, None

    # ── Manzil yig'ish bosqichi ───────────────────────────────────────────────
    # NOTE: price ichida history kerak emas — agent o'z state'ini boshqaradi
    classified = await classify_intent(query, lang)
    detected   = classified.get("intent", "price")
    det_lang   = classified.get("lang", lang)
    if det_lang in ("uz", "ru", "en"):
        lang = det_lang

    print(f"[PriceMultiTurn] classifier → intent={detected}")

    # ── TASK INTERRUPT ────────────────────────────────────────────────────────
    # Haqiqiy FAQ/tracking/location savoli (price slot javobi EMAS) kelsa —
    # price'ni PAUSE qilamiz (agent sessiyasi saqlanadi → keyin resume) va
    # boshqa intentga o'tkazamiz. Bu calculator'ning "qamab qo'yishini" oldini oladi.
    _INTERRUPT_INTENTS = {"faq", "tracking", "location"}
    if detected in _INTERRUPT_INTENTS and not _looks_like_price_slot(query):
        print(f"[PriceMultiTurn] INTERRUPT → '{detected}' (price paused, sessiya saqlandi)")
        _STATE_MGR.pause(user_key, reason_intent=detected)
        _clear_conv_state(user_key)
        # DIQQAT: agent.clear() chaqirilmaydi — yig'ilgan from/to/weight resume uchun qoladi
        return None, False, detected

    # ── Aks holda: price davom etadi (slot kiritildi / tasdiq / offtopic) ──────
    agent_response = await agent.run(
        user_key=user_key,
        query=query,
        lang=lang,
        turn=current_turn,
    )
    if agent_response.needs_more and agent_response.new_conv_state:
        _set_conv_state(user_key, agent_response.new_conv_state)
        _STATE_MGR.set_active(user_key, "price", agent_response.new_conv_state)
    else:
        # Hisob TUGADI → barcha holatni yopamiz (eski paused price resume bo'lmasligi uchun)
        _clear_conv_state(user_key)
        _STATE_MGR.close(user_key)
        agent.clear(user_key)
    await _save(user_key, query, agent_response.answer, intent="price")
    return agent_response.answer, True, None


# ─── LOCATION MULTI-TURN ───────────────────────────────────────────────────────

_SHORT_REPLY_RE = re.compile(
    r"^\s*(ha|yo'q|yoq|ok|okay|да|нет|yes|no|sure|fine|albatta|xo'p|хорошо)\s*$",
    re.I | re.U,
)

def _enrich_location(conv_state: Optional[dict], result: dict, query: str = "") -> dict:
    if not conv_state or conv_state.get("intent") != "location":
        return result
    if _SHORT_REPLY_RE.match(query):
        return result
    has_new_info = (
        result.get("form_query")
        or result.get("lat") is not None
        or result.get("city_hint")
    )
    if has_new_info:
        return result
    enriched = dict(result)
    if conv_state.get("form_query"):
        enriched["form_query"] = conv_state["form_query"]
        print(f"[MultiTurn/Location] form_query: {enriched['form_query']!r}")
    if conv_state.get("lat") is not None:
        enriched["lat"] = conv_state["lat"]
        enriched["lng"] = conv_state["lng"]
    return enriched


# ─── HISTORY ENDPOINT ──────────────────────────────────────────────────────────

@app.get("/api/history/", response_model=HistoryResponse)
async def get_chat_history(req: Request, browser_id: Optional[str] = None):
    user_key = _user_key(req, browser_id)
    messages = await get_history(user_key, limit=40)
    return HistoryResponse(
        messages=[HistoryMessage(**m) for m in messages],
        user_key=user_key,
    )


# ─── MAIN QUERY ────────────────────────────────────────────────────────────────

@app.post("/api/query/", response_model=QueryResponse)
@limiter.limit("20/minute")
async def api_query(req_body: QueryRequest, request: Request):
    query    = req_body.query.strip()
    lang     = _validate_lang(req_body.lang)
    user_key = _user_key(request, req_body.browser_id)

    # ── Global cache ──────────────────────────────────────────────────────────
    _has_coords = bool(re.search(r"-?\d{1,3}\.\d{3,8}.*,.*-?\d{1,3}\.\d{3,8}", query))
    if not _has_coords:
        global_cached = await get_global_cache(query)
        if global_cached:
            await _save(user_key, query, global_cached)
            return QueryResponse(
                answer=global_cached,
                cached=True,
                session_id=user_key,
                intent="cached",
                lang=lang,
            )

    conv_state   = _get_conv_state(user_key)
    chat_history = _history_get(user_key)

    # ── Suhbat boshqaruvi: aktiv / pauza / oxirgi mavzu (qisqa follow-up uchun) ──
    # Tartib: joriy active state → manager active → manager paused → eslab qolingan oxirgi mavzu
    _active_hint = (
        (conv_state.get("intent") if conv_state else None)
        or _STATE_MGR.active_intent(user_key)
        or _STATE_MGR.paused_intent(user_key)
        or get_last_intent(user_key)
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # YAGONA DISPATCH — qaror bitta joyda (chalkashlikni oldini oladi)
    #   • DECLINE  → hamma narsani yop, muloyim xayrlash
    #   • ACCEPT   → taklif berilgan intentni BOSHIDAN boshla
    #   • NEW_TASK → taklif bo'lgan bo'lsa eski intentni TOZALA, erkin qayta tasnifla
    # Bu klassifikatsiya va price-multiturn'dan OLDIN ishlaydi.
    # ═══════════════════════════════════════════════════════════════════════════
    _last_bot      = _last_bot_msg(chat_history)
    # Taklif belgisi: alohida xotiradan (history'dan emas — u strip qilinadi) + zaxira sifatida history
    _pending_offer = get_offer(user_key)
    clear_offer(user_key)   # iste'mol qilamiz; agar bu turn yana taklif qilsa, qayta o'rnatiladi
    _offer_intent  = _pending_offer or (get_last_intent(user_key) if _bot_made_offer(_last_bot) else None)
    _offer_pending = bool(_offer_intent)
    _has_active    = bool(conv_state) or _STATE_MGR.active_intent(user_key) or _STATE_MGR.paused_intent(user_key)

    # 1) DECLINE — faol vazifa yoki taklif bo'lsa, aniq rad → yopamiz
    if (_offer_pending or _has_active) and is_clear_decline(query):
        print(f"[Dispatch] DECLINE '{query}' → yopildi")
        _close_all(user_key)
        msg = _DECLINE_MSG.get(lang, _DECLINE_MSG["uz"])
        await _save(user_key, query, msg)
        return QueryResponse(answer=msg, cached=False, session_id=user_key,
                             intent="offtopic", lang=lang)

    # 2) ACCEPT — taklifga aniq rozilik → o'sha intentni boshidan boshlaymiz
    if _offer_pending and is_clear_accept(query):
        if _offer_intent in ("price", "location", "tracking"):
            print(f"[Dispatch] ACCEPT '{query}' → '{_offer_intent}' boshidan")
            _close_all(user_key)
            if _offer_intent == "price":
                _set_conv_state(user_key, {"intent": "price", "turn": 1})
                _STATE_MGR.set_active(user_key, "price", {"turn": 1})
            elif _offer_intent == "location":
                _set_conv_state(user_key, {"intent": "location", "form_query": None,
                                           "city_hint": None, "lat": None, "lng": None})
                _STATE_MGR.set_active(user_key, "location", {"form_query": None,
                                      "city_hint": None, "lat": None, "lng": None})
            remember_entities(user_key, intent=_offer_intent)
            prompt = _RESTART_PROMPT[_offer_intent].get(lang, _RESTART_PROMPT[_offer_intent]["uz"])
            await _save(user_key, query, prompt)
            return QueryResponse(answer=prompt, cached=False, session_id=user_key,
                                 intent=_offer_intent, lang=lang)

    # 3) NEW_TASK — taklif bor edi, lekin javob ha/yo'q emas → eski intentni TOZALA
    #    (intent ichida qamalib qolmaslik uchun) va erkin qayta tasnifla.
    if _offer_pending:
        print(f"[Dispatch] NEW_TASK '{query}' → eski intent tozalandi, erkin tasnif")
        _close_all(user_key)
        conv_state   = None
        _active_hint = None

    # ═══════════════════════════════════════════════════════════════════════════
    # PAUSED STATE RESUME CHECK
    # ═══════════════════════════════════════════════════════════════════════════

    is_price_active = conv_state and conv_state.get("intent") == "price"

    if not is_price_active:
        # v5.0: Classify intent history bilan (multi-turn)
        pre_classified = await classify_intent(
            query, lang=lang, history=chat_history, active_intent=_active_hint,
        )
        current_intent_pre = pre_classified.get("intent", "faq")

        # ── RESUME CHECK ──────────────────────────────────────────────────────
        resume: ResumeResult = _STATE_MGR.check_resume(
            user_key=user_key,
            current_intent=current_intent_pre,
            lang=lang,
        )

        if resume.should_resume:
            paused_data = resume.paused_data
            paused_intent = resume.paused_intent
            print(f"[Resume] {paused_intent} resume qilinmoqda | user={user_key[:8]}")

            if paused_intent == "price":
                saved_lang = paused_data.get("lang")
                if saved_lang in ("uz", "ru", "en"):
                    lang = saved_lang

                restored_conv_state = {"intent": "price", **paused_data}
                _set_conv_state(user_key, restored_conv_state)
                _STATE_MGR.resume(user_key)

                agent = _get_agents().get("price")
                current_turn = paused_data.get("turn", 1)
                agent_response = await agent.run(
                    user_key=user_key,
                    query=query,
                    lang=lang,
                    turn=current_turn,
                )
                if agent_response.needs_more and agent_response.new_conv_state:
                    _set_conv_state(user_key, agent_response.new_conv_state)
                    _STATE_MGR.set_active(user_key, "price", agent_response.new_conv_state)
                else:
                    _clear_conv_state(user_key)
                    _STATE_MGR.close(user_key)
                    agent.clear(user_key)

                await _save(user_key, query, agent_response.answer, intent="price")
                return QueryResponse(
                    answer=agent_response.answer,
                    cached=agent_response.cached,
                    session_id=user_key,
                    intent="price",
                    lang=lang,
                )

            _STATE_MGR.resume(user_key)

        else:
            if resume.paused_intent:
                _STATE_MGR.decrement_paused(user_key)

    # ═══════════════════════════════════════════════════════════════════════════
    # PRICE MULTI-TURN
    # ═══════════════════════════════════════════════════════════════════════════
    if conv_state and conv_state.get("intent") == "price":
        answer, should_return, fallback_intent = await _handle_price_multiturn(
            query=query,
            lang=lang,
            user_key=user_key,
            conv_state=conv_state,
        )

        if should_return:
            return QueryResponse(
                answer=answer,
                cached=False,
                session_id=user_key,
                intent="price",
                lang=lang,
            )

        intent = fallback_intent
        # v5.0: history bilan
        result = await classify_intent(query, lang=lang, history=chat_history, active_intent=_active_hint)
        lang   = result.get("lang", lang) if result.get("lang") in ("uz", "ru", "en") else lang

    else:
        # ═══════════════════════════════════════════════════════════════════════
        # ODDIY HOLAT
        # ═══════════════════════════════════════════════════════════════════════
        if not is_price_active and 'pre_classified' in dir():
            result = pre_classified
            intent = current_intent_pre
            lang   = result.get("lang", lang) if result.get("lang") in ("uz", "ru", "en") else lang
        else:
            # v5.0: history bilan
            result = await classify_intent(query, lang=lang, history=chat_history, active_intent=_active_hint)
            intent = result["intent"]
            lang   = result.get("lang", lang) if result.get("lang") in ("uz", "ru", "en") else lang

    # ── Result maydonlari ─────────────────────────────────────────────────────
    try:
        barcode    = result.get("barcode")
        form_query = result.get("form_query")
        city_hint  = result.get("city_hint")
        lat        = result.get("lat")
        lng        = result.get("lng")
    except (NameError, AttributeError):
        barcode = form_query = city_hint = lat = lng = None

    print(f"[Classifier] intent={intent} | form_query={form_query!r} | city_hint={city_hint!r} | lang={lang}")

    if intent != "location":
        _clear_conv_state(user_key)

    # ── Location multi-turn ───────────────────────────────────────────────────
    if intent == "location":
        enriched = _enrich_location(conv_state, result, query)
        form_query = enriched.get("form_query")
        city_hint = enriched.get("city_hint")
        lat = enriched.get("lat")
        lng = enriched.get("lng")

    # ── Suhbat boshqaruvi: entitylarni va oxirgi mavzuni eslab qol ────────────
    # (offtopic/prohibited oxirgi mavzuni o'chirmaydi — entities.py ichida hal qilingan)
    remember_entities(
        user_key,
        intent=intent,
        barcode=barcode or barcode_from_history(chat_history),
        form_query=form_query,
        city_hint=city_hint,
        lat=lat,
        lng=lng,
    )

    # ── PAUSE CHECK ───────────────────────────────────────────────────────────
    if not is_price_active and _STATE_MGR.should_pause_active(user_key, intent):
        print(f"[Pause] active state pause qilinmoqda (sabab: {intent})")
        _STATE_MGR.pause(user_key, reason_intent=intent)

    # ── Resume hint ───────────────────────────────────────────────────────────
    resume_hint_to_append: Optional[str] = None
    try:
        if not resume.should_resume and resume.resume_hint:
            resume_hint_to_append = resume.resume_hint
    except NameError:
        pass

    # ═══════════════════════════════════════════════════════════════════════════
    # INTENT HANDLERLARI
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Offtopic / prohibited ─────────────────────────────────────────────────
    if intent in ("offtopic", "prohibited"):
        cached = await get_global_cache(query)
        if cached:
            await _save(user_key, query, cached)
            return QueryResponse(
                answer=cached,
                cached=True,
                session_id=user_key,
                intent=intent,
                lang=lang,
            )

        answer = await _handle_offtopic(query, intent, lang, history=chat_history)
        answer = _append_hint(answer, resume) if 'resume' in dir() else answer
        await _save(user_key, query, answer)
        # MUHIM: offtopic javoblarni global cache'ga SAQLAMANG
        # Chunki "tentakmisan" → "salom" kabi shaxsiy/kontekstli savollarga
        # har foydalanuvchi uchun boshqacha javob kerak
        # if intent == "offtopic":
        #     await set_global_cache(query, answer, intent)
        return QueryResponse(
            answer=answer,
            cached=False,
            session_id=user_key,
            intent=intent,
            lang=lang,
        )

    # ── FAQ ───────────────────────────────────────────────────────────────────
    if intent == "faq":
        cached = await get_global_cache(query)
        if cached:
            await _save(user_key, query, cached)
            return QueryResponse(
                answer=cached,
                cached=True,
                session_id=user_key,
                intent=intent,
                lang=lang,
            )

        cached_ans = _cache_get("faq", query, lang)
        if cached_ans:
            await _save(user_key, query, cached_ans)
            return QueryResponse(
                answer=cached_ans,
                cached=True,
                session_id=user_key,
                intent=intent,
                lang=lang,
            )

        faq_conv_state = conv_state if (conv_state and conv_state.get("intent") == "faq") else None
        # v5.0: history bilan
        search_result = await SearchService.get().search(
            query=query, lang=lang, conv_state=faq_conv_state,
            history=chat_history,
        )
        answer = search_result.get("answer")
        if not answer:
            answer = await _handle_faq(query, lang, history=chat_history)

        new_faq_state = search_result.get("new_conv_state", {})
        new_faq_state["intent"] = "faq"
        _set_conv_state(user_key, new_faq_state)
        _STATE_MGR.set_active(user_key, "faq", new_faq_state)

        _cache_set("faq", query, lang, answer)

        # Follow-up: kontekstli savolni LLM/search javobining O'ZI yozadi.
        # Shablon faq_hook/💡 hint qo'shilmaydi — faqat (kerak bo'lsa, bir martalik)
        # resume hint, u ham javob savol bilan tugamagan bo'lsa.
        answer = _append_hint(answer, resume) if 'resume' in dir() else answer

        await _save(user_key, query, answer)
        await set_global_cache(query, answer, "faq")
        return QueryResponse(
            answer=answer,
            cached=False,
            session_id=user_key,
            intent=intent,
            lang=lang,
        )

    # ── Location ──────────────────────────────────────────────────────────────
    if intent == "location":
        # Manzil/koordinata umuman yo'q bo'lsa (masalan "ha" javobidan keyin) —
        # fallback bermaymiz, balki manzilni SO'RAYMIZ (mavzu davom etadi).
        if not (form_query or city_hint or (lat is not None)):
            ask_loc = {
                "uz": ("Albatta! Qaysi hudud yoki manzil bo'yicha pochta bo'limini "
                       "topib beray? Manzilni yozing yoki joylashuvingizni yuboring."),
                "ru": ("Конечно! По какому району или адресу найти отделение? "
                       "Напишите адрес или отправьте геолокацию."),
                "en": ("Sure! Which area or address should I search for a post office? "
                       "Type the address or share your location."),
            }.get(lang, "Qaysi manzil bo'yicha qidiray?")
            _set_conv_state(user_key, {
                "intent": "location", "form_query": None,
                "city_hint": None, "lat": None, "lng": None,
            })
            _STATE_MGR.set_active(user_key, "location", {
                "form_query": None, "city_hint": None, "lat": None, "lng": None,
            })
            await _save(user_key, query, ask_loc)
            return QueryResponse(
                answer=ask_loc, cached=False, session_id=user_key,
                intent="location", lang=lang,
            )

        cached_ans = _cache_get("location", query, lang) if not lat else None
        if not cached_ans and not lat:
            cached_ans = await get_global_cache(query)

        if cached_ans:
            cached_ans = _force_location_hook(cached_ans, lang)
            _set_conv_state(user_key, {
                "intent": "location", "form_query": form_query,
                "city_hint": city_hint, "lat": lat, "lng": lng,
            })
            _STATE_MGR.set_active(user_key, "location", {
                "form_query": form_query, "city_hint": city_hint,
                "lat": lat, "lng": lng,
            })
            await _save(user_key, query, cached_ans, intent="location")
            return QueryResponse(
                answer=cached_ans,
                cached=True,
                session_id=user_key,
                intent=intent,
                lang=lang,
            )

        from location.handler import handle as location_handle
        # v5.0: history bilan
        answer = await location_handle(
            query=query, form_query=form_query, city_hint=city_hint,
            lang=lang, lat=lat, lng=lng,
            history=chat_history,
        )

        # Hook'ni faqat-location taklifiga almashtiramiz (narx/tracking taklif qilmaydi)
        answer = _force_location_hook(answer, lang)

        _set_conv_state(user_key, {
            "intent": "location", "form_query": form_query,
            "city_hint": city_hint, "lat": lat, "lng": lng,
        })
        _STATE_MGR.set_active(user_key, "location", {
            "form_query": form_query, "city_hint": city_hint,
            "lat": lat, "lng": lng,
        })

        answer = _append_hint(answer, resume) if 'resume' in dir() else answer

        await _save(user_key, query, answer, intent="location")
        return QueryResponse(
            answer=answer,
            cached=False,
            session_id=user_key,
            intent=intent,
            lang=lang,
        )

    # ── Tracking ──────────────────────────────────────────────────────────────
    if intent == "tracking":
        from base_agent import AgentContext
        agent = _get_agents().get("tracking")
        # ── Suhbat boshqaruvi: trek-kod bir marta berilgan bo'lsa qayta so'ramaymiz ──
        # Tartib: joriy xabar → eslab qolingan → suhbat tarixi.
        # Agent o'zgarmaydi: u ctx.extra["barcode"] ni avval o'qiydi.
        eff_barcode = (
            barcode
            or get_remembered_barcode(user_key)
            or barcode_from_history(chat_history)
        )
        if eff_barcode and eff_barcode != barcode:
            print(f"[Tracking/ctx] trek-kod kontekstdan tiklandi: {eff_barcode}")
        # v5.0: history extra orqali uzatiladi
        ctx   = AgentContext(
            query=query, lang=lang, user_key=user_key,
            intent=intent, conv_state=conv_state,
            extra={"barcode": eff_barcode, "history": chat_history},
        )
        agent_response = await agent.run(ctx)
        if agent_response.needs_more and agent_response.new_conv_state:
            _set_conv_state(user_key, agent_response.new_conv_state)
            _STATE_MGR.set_active(user_key, "tracking", agent_response.new_conv_state)
        else:
            _clear_conv_state(user_key)
            _STATE_MGR.close_active(user_key)

        answer = agent_response.answer
        answer = _append_hint(answer, resume) if 'resume' in dir() else answer

        await _save(user_key, query, answer, intent="tracking")
        return QueryResponse(
            answer=answer,
            cached=agent_response.cached,
            session_id=user_key,
            intent=intent,
            lang=lang,
        )

    # ── Price (yangi sessiya) ─────────────────────────────────────────────────
    if intent == "price":
        agent          = _get_agents().get("price")
        agent_response = await agent.run(
            user_key=user_key, query=query, lang=lang, turn=1,
        )
        if agent_response.needs_more and agent_response.new_conv_state:
            _set_conv_state(user_key, agent_response.new_conv_state)
            _STATE_MGR.set_active(user_key, "price", agent_response.new_conv_state)
        else:
            _clear_conv_state(user_key)
            _STATE_MGR.close(user_key)
            agent.clear(user_key)

        await _save(user_key, query, agent_response.answer, intent="price")
        return QueryResponse(
            answer=agent_response.answer,
            cached=agent_response.cached,
            session_id=user_key,
            intent=intent,
            lang=lang,
        )

    # ── Fallback ──────────────────────────────────────────────────────────────
    answer = await _handle_faq(
        query, lang, note="Ushbu xizmat hozircha ishlab chiqilmoqda.",
        history=chat_history,
    )
    await _save(user_key, query, answer)
    return QueryResponse(
        answer=answer,
        cached=False,
        session_id=user_key,
        intent=intent,
        lang=lang,
    )


# ─── FAQ FALLBACK ──────────────────────────────────────────────────────────────

_FAQ_SYS = {
    "uz": (
        "Sen Pochtachi Bek — UzPostning rasmiy yordamchi botisan.\n"
        "Mijozlar bilan tajribali, samimiy va xushmuomala operator kabi muloqot qil.\n\n"
        
            "ASOSIY VAZIFA:\n"
            "- Mijozning savoliga to‘liq, aniq va tushunarli javob ber.\n"
            "- Faqat ishonchli va mavjud ma’lumotlardan foydalan.\n"
            "- Agar aniq ma’lumot mavjud bo‘lmasa, buni ochiq ayt.\n\n"
        
            "JAVOB BERISH USLUBI:\n"
            "- Tabiiy, iliq va insoniy uslubda yoz.\n"
            "- Javoblar ravon, mazmunli va to‘liq bo‘lsin — odatda 4–6 ta gap.\n"
            "- Mavzuni mijozga tushunarli qilib, zarur tafsilotlar bilan ochib ber.\n"
            "- Keraksiz rasmiy yoki robotga o‘xshash iboralardan foydalanma.\n"
            "- Follow-up savol har doim emas — faqat foydali bo‘lsa va mijozning\n"
            "  aynan savolidan kelib chiqsa bitta qo‘sh; aks holda berma.\n\n"
        
            "MISOLLAR:\n"
            "- Jo‘natmangizni kuzatib beraymi?\n"
            "- Sizga yaqin pochta bo‘limini topib beraymi?\n"
            "- Yetkazib berish narxini hisoblab beraymi?\n\n"
        
            "TAQIQLANADI:\n"
            "- Emoji ishlatish.\n"
            "- Markdown formatlashdan foydalanish.\n"
            "- O‘ylab topilgan yoki tasdiqlanmagan ma’lumotlarni berish.\n"
            "- Ma’nosiz yoki takroriy javoblar yozish.\n"
            "- Har bir javobda bir xil qolipdagi iboralarni takrorlash.\n\n"
        
            "1165 raqami yoki uz.post saytini faqat tariflar, manzillar, "
            "ish vaqti yoki boshqa aniq ma’lumotlar mavjud bo‘lmagan holatlarda tavsiya qil."
        ),
    "ru": (
        'Ты Pochtachi Bek — официальный бот-помощник UzPost.\n'
        'Говори как опытный, внимательный и проактивный оператор.\n\n'

        'СТИЛЬ ОТВЕТА:\n'
        '- Отвечай полно и точно на вопрос клиента (4-6 предложений).\n'
        '- Follow-up НЕ ВСЕГДА — добавляй вопрос, только если он реально помогает\n'
        '  и вытекает из КОНКРЕТНОГО вопроса клиента, а не шаблон.\n'
        '- Если вопрос закрыт и ответ полный — без лишнего вопроса.\n'
        '- Пиши естественно и по-человечески — не как робот.\n'
        '- Текстом, не списком.\n\n'

        'ЗАПРЕЩЕНО:\n'
        '- Emoji и markdown.\n'
        '- Шаблонные фразы типа "Если есть вопросы, обращайтесь".\n'
        '- Повторять один и тот же вопрос в конце каждого ответа.\n'
        '- Придумывать информацию.\n'
        '- 1165 и uz.post — ТОЛЬКО когда нет точных данных о тарифах,\n'
        '  адресах, часах работы — не в каждом ответе!'
    ),
    "en": (
        'You are Pochtachi Bek — official UzPost assistant bot.\n'
        'Be experienced, warm, and proactive — like a real person.\n\n'

        'RESPONSE STYLE:\n'
        '- Answer the customer\'s question fully and clearly (4-6 sentences).\n'
        '- A follow-up is NOT ALWAYS needed — add one only if it truly helps\n'
        '  and follows from the customer\'s SPECIFIC question, not a template.\n'
        '- If the question is closed and the answer is complete — no extra question.\n'
        '- Write naturally and humanly — not like a robot.\n'
        '- Prose, not bullet points.\n\n'

        'PROHIBITED:\n'
        '- Emoji and markdown.\n'
        '- Generic phrases like "Feel free to ask anything".\n'
        '- Repeating the same question at the end of every answer.\n'
        '- Inventing information.\n'
        '- Mention 1165 or uz.post ONLY when specific data (rates, addresses,\n'
        '  hours) is unavailable — not in every response!'
    ),
}

_NO_LLM = {
    "uz": "Kechirasiz, so‘rovni qayta ishlashda xatolik yuz berdi.\n1165 | https://uz.post",
    "ru": "Извините, ошибка.\n1165 | https://uz.post",
    "en": "Sorry, error.\n1165 | https://uz.post",
}


async def _handle_faq(
    query: str,
    lang: str,
    note: str = "",
    history: list[dict] | None = None,
) -> str:
    """v5.0: history bilan multi-turn javob"""
    if _FAQ_LLM is None:
        return _NO_LLM.get(lang, "1165 | https://uz.post")

    from langchain_core.messages import AIMessage

    extra = f"\nIzoh: {note}" if note else ""
    messages = [SystemMessage(content=_FAQ_SYS.get(lang, _FAQ_SYS["uz"]))]

    # v5.0: history qo'shamiz (oxirgi 4 xabar = 2 turn)
    if history:
        for msg in history[-4:]:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if msg["role"] == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query + extra))

    try:
        resp = await _FAQ_LLM.ainvoke(messages)
        return resp.content.strip()
    except Exception as e:
        print(f"[FAQ LLM] {e}")
        return _NO_LLM.get(lang, "1165 | https://uz.post")


# ─── OFFTOPIC ──────────────────────────────────────────────────────────────────

_OFFTOPIC_SYS = {
    "uz": (
        "Sen Pochtachi Bek — UzPost xizmatlari bo‘yicha yordamchi botsan. "
        "Mijozlar bilan tajribali, samimiy va xushmuomala operator kabi muloqot qil.\n\n"
        
            "SUHBAT QOIDALARI:\n"
        
            "1. Agar mijoz salomlashsa, hol-ahvol so‘rasa yoki tanishmoqchi bo‘lsa:\n"
            "   - Samimiy va iliq tarzda javob ber.\n"
            "   - O‘zingni Pochtachi Bek sifatida tanishtir.\n"
            "   - Javob oxirida UzPost xizmatlariga oid bitta aniq taklif yoki savol ber.\n"
            "   Masalan:\n"
            "   • Jo‘natmangizni kuzatib beraymi?\n"
            "   • Sizga yaqin pochta bo‘limini topib beraymi?\n"
            "   • Yetkazib berish narxini hisoblab beraymi?\n"
            "   • Qaysi xizmat bo‘yicha yordam kerak?\n\n"
        
            "2. Agar mijozning savoli UzPost xizmatlariga aloqador bo‘lmasa:\n"
            "   - Buni muloyim tarzda tushuntir.\n"
            "   - Suhbatni UzPost xizmatlariga yo‘naltir.\n"
            "   - Javob oxirida UzPost xizmatlariga oid bitta aniq taklif ber.\n\n"
        
            "JAVOB BERISH TALABLARI:\n"
            "- Javoblar tabiiy, iliq va jonli bo‘lsin — xuddi tajribali operator gapirayotgandek.\n"
            "- Odatda 4–6 ta gap bilan, mazmunli javob ber. Quruq bir-ikki gap bilan cheklanma.\n"
            "- Mijozning kayfiyati va savoliga moslashib, samimiy ohangda yoz.\n"
            "- Kerak bo‘lsa, UzPost qanday yordam bera olishini qisqacha tushuntir.\n"
            "- Har bir javob oxirida bitta aniq, foydali savol yoki taklif bo‘lsin.\n"
            "- O‘zingni sun’iy intellekt, AI yoki til modeli deb atama.\n"
            "- Noto‘g‘ri yoki o‘ylab topilgan ma’lumotlarni bermagin.\n"
            "- Emoji va Markdown ishlatma.\n"
            "- Zarurat bo‘lsa, 1165 aloqa markazi raqamini tavsiya qil."
        ),
    "ru": (
        "Ты Pochtachi Bek — оператор колл-центра UzPost. "
        "Говори как опытный, тёплый и проактивный оператор.\n\n"

        "ВЕДЕНИЕ ДИАЛОГА:\n"
        "Если клиент здоровается или хочет познакомиться:\n"
        "  → Ответь тепло, представься\n"
        "  → ВСЕГДА предложи что-то конкретное:\n"
        '     "Отследить ваше отправление?"\n'
        '     "Найти ближайшее отделение?"\n'
        '     "Рассчитать стоимость доставки?"\n'
        '     "Какая услуга вам нужна?"\n\n'

        "Если вопрос не о UzPost:\n"
        "  → Вежливо объясни, переведи разговор на услуги UzPost\n"
        "  → Предложи что-то конкретное\n\n"

        "ПРАВИЛА:\n"
        "- Не называй себя ИИ\n"
        "- 4-6 предложений, живо и тепло — как опытный оператор. Не ограничивайся сухими 2 фразами\n"
        "- Подстраивайся под настроение и вопрос клиента\n"
        "- В конце всегда конкретный полезный вопрос или предложение\n"
        "- Не придумывай информацию\n"
        "- Без emoji и markdown\n"
        "- При необходимости: 1165"
    ),

    "en": (
        "You are Pochtachi Bek — UzPost call center operator. "
        "Be warm, experienced, and proactive.\n\n"

        "CONVERSATION FLOW:\n"
        "When customer greets or wants to chat:\n"
        "  → Respond warmly, introduce yourself\n"
        "  → ALWAYS offer something specific:\n"
        '     "Track your shipment?"\n'
        '     "Find your nearest post office?"\n'
        '     "Calculate delivery costs?"\n'
        '     "Which service works best for you?"\n\n'

        "If question is not about UzPost:\n"
        "  → Politely explain, redirect to UzPost services\n"
        "  → Offer something specific\n\n"

        "RULES:\n"
        "- Never call yourself AI\n"
        "- 4-6 sentences, warm and lively — like a seasoned operator. Don't settle for a dry one-liner\n"
        "- Adapt to the customer's mood and question\n"
        "- Always end with a specific, helpful question or offer\n"
        "- Never invent information\n"
        "- No emoji or markdown\n"
        "- If needed: 1165"
    ),
}

_PROHIBITED = {
    "uz": "Bunday turdagi buyumlarni pochta orqali yuborish qonunchilikka muvofiq taqiqlangan.\nSizga UzPost xizmatlari bo‘yicha boshqa masalada yordam bera olaman..\n1165 | https://uz.post",
    "ru": "Пересылка таких предметов запрещена законом.\nЕсли есть другие вопросы — готов помочь.\n1165 | https://uz.post",
    "en": "Sending such items is prohibited by law.\nIf you have other questions, I'm happy to help.\n1165 | https://uz.post",
}

_OFFTOPIC_FALLBACK = {
    "uz": (
        "Salom! Men Pochtachi Bek — UzPost rasmiy yordamchi botiman.\n"
        "Quyidagi xizmatlarda yordam bera olaman:\n"
        "jo‘natmalarni kuzatish, yetkazib berish narxini hisoblash, "
        "eng yaqin pochta bo‘limlarini topish hamda UzPost xizmatlari bo‘yicha ma'lumot berish.\n"
        "Qanday yordam kerak?"
    ),
    "ru": (
        "Привет! Я Pochtachi Bek — официальный бот-помощник UzPost.\n"
        "Могу помочь с: отслеживанием посылок, расчётом стоимости доставки, "
        "поиском ближайшего отделения и общими вопросами.\n"
        "Чем могу помочь?"
    ),
    "en": (
        "Hello! I'm Pochtachi Bek — official UzPost assistant bot.\n"
        "I can help with: shipment tracking, delivery cost calculation, "
        "finding the nearest post office, and general questions.\n"
        "How can I help you?"
    ),
}


async def _handle_offtopic(query: str, intent: str, lang: str, history: list[dict] | None = None) -> str:
    if intent == "prohibited":
        return _PROHIBITED.get(lang, "1165 | https://uz.post")

    if _OFFTOPIC_LLM is None:
        return _OFFTOPIC_FALLBACK.get(lang, "1165 | https://uz.post")

    from langchain_core.messages import AIMessage

    messages = [
        SystemMessage(content=_OFFTOPIC_SYS.get(lang, _OFFTOPIC_SYS["uz"])),
    ]

    # history qo'shamiz — context uchun (oxirgi 4 xabar)
    if history:
        for msg in history[-4:]:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if msg["role"] == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query))

    try:
        resp = await _OFFTOPIC_LLM.ainvoke(messages)
        return resp.content.strip()
    except Exception as e:
        print(f"[Offtopic LLM] {e}")
        return _OFFTOPIC_FALLBACK.get(lang, "1165 | https://uz.post")


# ─── HEALTH ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "faq_llm_ready": _FAQ_LLM is not None,
        "state_mgr":     len(_STATE_MGR._entries),
    }
