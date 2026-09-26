"""
intent_classifier.py
=====================
O'ZGARISHLAR (avvalgi versiyaga nisbatan):
  1. _llm_classify endi history qabul qiladi va LLM'ga messages format'ida uzatadi
  2. _CLASSIFY_SYSTEM ga "MULTI-TURN CONTEXT" bo'limi qo'shildi
  3. classify_intent endi `history` parametrini qabul qiladi
  4. LLM obyektlari endi singleton (har request da yangi yaratilmaydi)
  5. Qisqa javoblar uchun maxsus logika — last_bot_message bilan resolve
"""
from __future__ import annotations

import os
import re
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

from conversation.entities import is_greeting, is_identity

load_dotenv()

# Mavzu (topic) intentlari — qisqa follow-up shu intentlardan birini davom ettiradi
_VALID_TOPIC = {"tracking", "price", "location", "faq"}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL   = os.getenv("GROQ_MODEL_PREMIUM", "llama-3.3-70b-versatile")


# ─── Til aniqlash (o'zgarmagan) ───────────────────────────────────────────────

_CYRILLIC = re.compile(r"[а-яё]", re.I | re.U)
_UZ_HINTS = re.compile(
    r"\b(salom|rahmat|qanday|qachon|nima|qaerda|kerak|jo'?nat|"
    r"pochta|kuzat|bormi|necha|qancha|posilka|yetkaz|qadoq|olib|topshir|"
    r"chi|man|miz|siz|dan|ga|da|ning|gan|moqchi|olmoq|kerak)\b",
    re.I | re.U,
)
_EN_HINTS = re.compile(
    r"\b(hello|hi|please|track|parcel|shipping|delivery|nearest|office|"
    r"how|where|when|what|why|need|want|send|cost)\b",
    re.I,
)


def detect_lang(text: str) -> str:
    """Sodda til detektori: kirill→ru, en signallar→en, default→uz."""
    if not text:
        return "uz"

    if _CYRILLIC.search(text):
        return "ru"

    has_en = bool(_EN_HINTS.search(text))
    has_uz = bool(_UZ_HINTS.search(text))

    if has_uz:
        return "uz"
    if has_en and not has_uz:
        return "en"

    # Fallback: lingua-detector (agar mavjud bo'lsa)
    try:
        from lingua import Language, LanguageDetectorBuilder
        global _LINGUA
        if "_LINGUA" not in globals() or _LINGUA is None:
            _LINGUA = (
                LanguageDetectorBuilder
                .from_languages(Language.UZBEK, Language.ENGLISH, Language.RUSSIAN)
                .build()
            )
        detected = _LINGUA.detect_language_of(text)
        if detected:
            return {"UZBEK": "uz", "ENGLISH": "en", "RUSSIAN": "ru"}.get(detected.name, "uz")
    except Exception:
        pass

    return "uz"


# ─── Hard signal extractor'lar ───────────────────────────────────────────────

_BARCODE_RE = re.compile(r"\b([A-Z]{2}\d{9}[A-Z]{2})\b")


def _extract_barcode(text: str) -> str | None:
    m = _BARCODE_RE.search(text.upper())
    return m.group(1) if m else None


_LATLON_RE = re.compile(
    r"\(?\s*(-?\d{1,2}[.,]\d+)\s*[,;]\s*(-?\d{1,3}[.,]\d+)\s*\)?"
)


# ═══════════════════════════════════════════════════════════════════
# MULTI-TURN: "ha / ok / да" tasdig'ini oxirgi bot taklifiga bog'lash
# ═══════════════════════════════════════════════════════════════════
# Bot javobi oxirida "...narxini hisoblab beraymi?" kabi taklif beradi.
# Mijoz "ha" deganda LLM ba'zan offtopic qaytaradi → loop. Buni oldini
# olish uchun deterministik: oxirgi bot xabaridagi taklifni o'qib,
# to'g'ridan-to'g'ri o'sha intentga bog'laymiz.

_AFFIRM_RE = re.compile(
    r"^\s*(ha+|ha'?a|xo'?p|mayli|albatta|bo'?lad[ia]|davom\s*et\w*|"
    r"ok(ay)?|yes|sure|yeah|yep|yup|"
    r"да|давай|конечно|ага|угу|можно|хочу|продолж\w*)"
    # ixtiyoriy tasdiqlovchi yuklama: "ha kerak", "ha bor", "да нужно", "ha davom ettiramiz"
    r"(\s+(kerak\w*|bor|mayli|bo'?ladi|albatta|please|"
    r"нужно|надо|давай|продолж\w*|davom\s*et\w*|et\w*))?"
    r"\s*[.!,]*\s*$",
    re.I | re.U,
)

_NEGATE_RE = re.compile(
    r"^\s*(yo'?q|yoq|kerakmas|kerak\s*emas|shart\s*emas|"
    r"no|nope|nah|"
    r"нет|не\s*надо|не\s*нужно|незачем)\s*[.!,]*\s*$",
    re.I | re.U,
)

# Oxirgi bot xabaridagi taklif qaysi xizmatga oid ekanini aniqlovchi signallar
_OFFER_PRICE_RE = re.compile(
    r"(narx|hisobla|qancha\s*tur|qancha\s*bo'?l|to'?lov|tarif|"
    r"стоимост|рассчита|расчёт|расчет|цен|сколько\s*стоит|"
    r"price|cost|calculat|how\s*much)",
    re.I | re.U,
)
_OFFER_TRACK_RE = re.compile(
    r"(kuzat|trek|jo'?natma\w*\s*(qayer|holat|kuzat)|posilka\w*\s*qayer|"
    r"отслеж|трек|где\s*(моя\s*)?посылк|статус\s*отправлени|"
    r"track|where\s*is\s*(my\s*)?(parcel|package|shipment))",
    re.I | re.U,
)
_OFFER_LOC_RE = re.compile(
    r"(pochta\s*bo'?lim|yaqin\w*\s*pochta|pochtani\s*top|manzil|ish\s*vaqt|"
    r"отделени|ближайш\w*\s*(почт|отделени)|адрес|режим\s*работ|"
    r"post\s*office|nearest|branch|working\s*hours)",
    re.I | re.U,
)


def _last_assistant_msg(history: list[dict] | None) -> str:
    """History'dagi eng oxirgi bot (assistant) xabarini qaytaradi."""
    if not history:
        return ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return (msg.get("content") or "").strip()
    return ""


def _offer_intent_from_bot(bot_msg: str) -> str | None:
    """
    Bot xabaridagi taklifni o'qib, qaysi intentga oid ekanini aniqlaydi.

    Bir nechta taklif bo'lsa — oxirgi (CTA odatda gap oxirida) tanlanadi.
    Aniq bitta taklif topilmasa → None (LLM hal qiladi yoki aniqlik so'raladi).
    """
    if not bot_msg:
        return None
    matches: list[tuple[int, str]] = []
    for rx, intent in (
        (_OFFER_PRICE_RE, "price"),
        (_OFFER_TRACK_RE, "tracking"),
        (_OFFER_LOC_RE,   "location"),
    ):
        m = rx.search(bot_msg)
        if m:
            matches.append((m.start(), intent))
    if not matches:
        return None
    distinct = {i for _, i in matches}
    if len(distinct) == 1:
        return matches[0][1]
    # Bir nechta xil taklif — oxirgi joylashganini olamiz (yakuniy CTA)
    matches.sort()
    return matches[-1][1]


def _extract_latlon(text: str) -> tuple[float, float] | None:
    m = _LATLON_RE.search(text)
    if not m:
        return None
    try:
        lat = float(m.group(1).replace(",", "."))
        lng = float(m.group(2).replace(",", "."))
        if 37.0 <= lat <= 45.7 and 55.9 <= lng <= 73.2:
            return lat, lng
    except ValueError:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════
# LLM SINGLETONLAR (har request da yangi obyekt yaratilmaydi)
# ═══════════════════════════════════════════════════════════════════

_CLASSIFY_LLM: Optional[ChatGroq] = None
_LOCATION_LLM: Optional[ChatGroq] = None


def _get_classify_llm() -> ChatGroq:
    global _CLASSIFY_LLM
    if _CLASSIFY_LLM is None:
        _CLASSIFY_LLM = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=25,
            api_key=GROQ_API_KEY,
        )
    return _CLASSIFY_LLM


def _get_location_llm() -> ChatGroq:
    global _LOCATION_LLM
    if _LOCATION_LLM is None:
        _LOCATION_LLM = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=30,
            api_key=GROQ_API_KEY,
        )
    return _LOCATION_LLM


# ═══════════════════════════════════════════════════════════════════
# CLASSIFY SYSTEM PROMPT (history bilan ishlash uchun yangilangan)
# ═══════════════════════════════════════════════════════════════════

_CLASSIFY_SYSTEM = """You are the intent classifier for UzPost (Uzbekistan Post) chatbot.

Return EXACTLY ONE label only:
tracking | price | location | faq | offtopic | prohibited

========================
INTENT DEFINITIONS
========================

tracking — user's OWN shipment: status, location of parcel, barcode, delay, customs, SMS, lost, delivered.
Examples: "RF123456789UZ qayerda?", "pochta yetib keldimi?", "sms kelmadi", "jo'natma kechikdi"

price — SPECIFIC shipping COST calculation (needs route + weight or destination + weight).
Examples: "Buxorodan Navoiyga 2kg qancha?", "5kg EMS Moskva narxi"
NOT price: "tarif jadvali", "EMS narxlari", "qancha turadi o'zi" → those are faq

location — post office branch, address, working hours, services AT a branch, phone of branch.
Examples: "Olmazor pochta qayerda?", "yakshanba ishlaydimi?", "100070 ish vaqti"

faq — general info: services, rules, instructions, general tariffs, how-to.
Examples: "EMS nima?", "qanday jo'nataman?", "pasport kerakmi?", "qadoqlash qoidalari"

offtopic — non-UzPost: weather, food, politics, jokes, greetings, AI identity, casual chat.
Examples: "salom", "sen kimsan?", "bugun ob havo", "qandaysan"

prohibited — illegal items: drugs, weapons, explosives, counterfeit, gold export.
Examples: "qurol yuborsam bo'ladimi?", "narkotik jo'natish"

========================
MULTI-TURN CONTEXT RULES (CRITICAL)
========================

You will see PRIOR conversation messages before the current message.
Use them to resolve AMBIGUOUS or SHORT user replies.

RULE 1 — Short affirmations ("ha", "yo'q", "ok", "bor", "yes", "no", "да", "нет"):
  Look at the LAST assistant message. If it asked about a topic, the reply CONTINUES that topic.

  Example:
    Assistant: "Trek raqamingizni yuboring"
    User: "ha"           → tracking
    User: "bor"          → tracking

    Assistant: "Pochta bo'limini qidiruvni davom ettiramizmi?"
    User: "ha"           → location
    User: "davom ettir"  → location

    Assistant: "Boshqa savolingiz bormi?"
    User: "bor"          → look DEEPER in history — what was the previous topic? Return THAT.

RULE 2 — Continuation phrases ("davom et", "продолжи", "continue"):
  Always continue the PREVIOUS intent shown in history.

RULE 3 — Single ambiguous words ("nima", "kim", "qaerda", "qanday", "что", "где"):
  These are usually SUB-questions about the ongoing topic in history.
  - Discussing tracking → tracking
  - Discussing FAQ topic → faq
  - No clear context → faq

RULE 4 — Greetings + identity ("salom", "kimsan", "qandaysan", "isming nima"):
  ALWAYS return "offtopic". These are NEVER tracking/price/location/faq even mid-conversation.

RULE 5 — NEW topic words override history:
  - barcode (AB123456789CD) → tracking
  - coordinates (41.x, 69.x) → location
  - "narx", "qancha pul" with route → price
  - "yaqin pochta", "почта рядом" → location

RULE 6 — Topic switch detection:
  If user's current message clearly introduces a NEW topic (different domain words),
  classify by the NEW topic, NOT history. Example:
    Prior: discussing tracking
    User: "Olmazor pochta qayerda?" → location (NEW topic, override history)

========================
FOLLOW-UP RESOLUTION (DYNAMIC — use ACTIVE TOPIC)
========================
When an "ACTIVE TOPIC" is provided with the current message, it is the subject the
user is currently engaged in (their last ongoing intent).

If the current message is a SHORT or AMBIGUOUS continuation — a bare question word,
a fragment, or a follow-up such as "qachon?", "qayerda?", "qancha?", "unda?",
"yana?", "nega?", "bo'ldimi?", "kechikdimi?", "hali kelmadimi?", "shunaqami?",
"-mi/-chi" style questions, "when?", "where?", "how long?", "and then?",
"сколько ещё?", "потом?" — AND it does NOT clearly start a new domain AND is NOT a
greeting/identity line, then RETURN THE ACTIVE TOPIC. Do not reclassify it.

Override the ACTIVE TOPIC ONLY when the message clearly starts a new domain
(barcode → tracking, "X pochta qayerda" → location, weight+route → price,
"qanday jo'nataman" / "EMS nima" → faq) OR is a greeting/identity (→ offtopic).

When unsure between continuing the ACTIVE TOPIC and "faq", PREFER the ACTIVE TOPIC.

Reply with ONE WORD only. No explanation.

========================
SPECIAL CASE: ENGAGE DETECTION
========================
If the user says ONLY a short greeting/check-in with NO postal question:
  "salom", "assalomu alaykum", "hi", "hello", "привет", "qandaysan",
  "yaxshimisiz", "kimsan", "isming nima" → ALWAYS "offtopic"

If user says a short affirmation like "ha", "bor", "ok", "yes", "да" AND
the last assistant message ends with "💬" or "?" then continue THAT topic.

Reply with ONE WORD only."""


# ═══════════════════════════════════════════════════════════════════
# LLM classify (history bilan)
# ═══════════════════════════════════════════════════════════════════

async def _llm_classify(
    query:   str,
    history: list[dict] | None = None,
    active_intent: str | None = None,
) -> str | None:
    """
    LLM'ga messages format'ida history + joriy savolni uzatadi.
    history: [{"role": "user"|"assistant", "content": "..."}]
    active_intent: hozir faol/oxirgi mavzu — qisqa davomlar uchun yo'naltiruvchi.
    """
    try:
        llm = _get_classify_llm()

        messages = [SystemMessage(content=_CLASSIFY_SYSTEM)]

        # Oxirgi 2 ta xabar (1 turn) — chalkashlikni kamaytiradi (uzun tarix
        # LLM'ni eski mavzuga "yopishtirib" qo'yadi). Faqat eng yaqin kontekst.
        if history:
            for msg in history[-2:]:
                role = msg.get("role")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # Joriy savol (+ aktiv mavzu konteksti)
        hint = ""
        if active_intent in _VALID_TOPIC:
            hint = (
                f'\n[ACTIVE TOPIC = {active_intent}. If this message is a short or '
                f'ambiguous follow-up/continuation (not a clear new domain, not a '
                f'greeting/identity), you MUST answer "{active_intent}". '
                f'When unsure, prefer "{active_intent}".]'
            )
        messages.append(HumanMessage(content=f'Current message: "{query}"{hint}\nIntent:'))

        resp = await llm.ainvoke(messages)
        raw  = resp.content.strip().lower()
        word = re.sub(r"[^a-z]", "", raw.split()[0]) if raw.split() else ""
        valid = {"tracking", "price", "location", "faq", "offtopic", "prohibited"}
        return word if word in valid else None

    except Exception as e:
        print(f"[LLM/classify] Xato: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# LOCATION EXTRACTION (o'zgarmagan)
# ═══════════════════════════════════════════════════════════════════

_LOCATION_SYSTEM = """UzPost location extractor. Return exactly 2 lines:
MICRO: <MFY/street/massiv/mahalla/kvartal/mavze — specific place, with suffix if present>
CITY: <city or district name>
If absent → leave empty after colon.

Examples:
"G'ijduvon toshloq mfydan pochta?" → MICRO: Toshloq MFY\nCITY: G'ijduvon
"qarshi islom karimov ko'chasida"  → MICRO: Islom Karimov ko'chasi\nCITY: Qarshi
"yunusobod bodomzorda pochta"      → MICRO: Bodomzor\nCITY: Yunusobod
"samarqandda pochta"               → MICRO:\nCITY: Samarqand
"eng yaqin pochta qayerda"         → MICRO:\nCITY:"""


async def _llm_location(query: str) -> tuple[str | None, str | None]:
    try:
        llm  = _get_location_llm()
        resp = await llm.ainvoke([
            SystemMessage(content=_LOCATION_SYSTEM),
            HumanMessage(content=f'Message: "{query}"'),
        ])
        raw = resp.content.strip()

        micro = None
        city  = None
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("MICRO:"):
                v = line.split(":", 1)[1].strip()
                micro = v if v else None
            elif line.upper().startswith("CITY:"):
                v = line.split(":", 1)[1].strip()
                city = v if v else None
        return micro, city
    except Exception as e:
        print(f"[LLM/location] Xato: {e}")
        return None, None


# ═══════════════════════════════════════════════════════════════════
# ASOSIY ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

async def classify_intent(
    query:   str,
    lang:    str = "",
    history: list[dict] | None = None,
    active_intent: str | None = None,
) -> dict:
    """
    Asosiy intent classifier.

    Parameters
    ----------
    query    : foydalanuvchi savoli
    lang     : til ('uz'/'ru'/'en'). Bo'sh kelsa — detect_lang fallback
    history  : oxirgi xabarlar [{"role": "user"|"assistant", "content": str}]
               Multi-turn aniqlash uchun ishlatiladi.
    active_intent : main.py uzatadigan hozirgi faol / pauza qilingan / oxirgi
               mavzu. Qisqa follow-up xabarlarni shu mavzuda davom ettirish uchun.

    Returns
    -------
    dict: {intent, lang, barcode, form_query, city_hint, lat, lng}
    """
    if not lang or lang not in ("uz", "ru", "en"):
        lang = detect_lang(query)

    # ── Hard signallar (history kerak emas) ─────────────────
    barcode = _extract_barcode(query)
    if barcode:
        return _r("tracking", lang, barcode=barcode)

    coords = _extract_latlon(query)
    if coords:
        lat, lng = coords
        return _r("location", lang, lat=lat, lng=lng)

    # ── MULTI-TURN: "ha/ok/да" → oxirgi bot taklifiga bog'lash ──
    # Bu LLM'dan OLDIN ishlaydi — deterministik, ishonchli.
    # Mijoz qisqa tasdiq bersa ("ha", "albatta", "davom et"),
    # oxirgi bot xabaridagi taklif qaysi xizmat bo'lsa — o'shanga o'tamiz.
    if history and _AFFIRM_RE.match(query):
        last_bot = _last_assistant_msg(history)
        offered  = _offer_intent_from_bot(last_bot)
        if offered:
            print(f"[Classifier] affirm '{query}' → oxirgi taklif: {offered}")
            if offered == "location":
                # location uchun micro/city kerak — bu yerda yo'q, handler so'raydi
                return _r("location", lang)
            return _r(offered, lang)
        # Aniq taklif topilmasa, lekin aktiv mavzu bo'lsa — o'sha mavzuni davom ettiramiz
        if active_intent in _VALID_TOPIC:
            print(f"[Classifier] affirm '{query}' → aktiv mavzu davom: {active_intent}")
            return _r(active_intent, lang)
        # Aks holda — LLM'ga tashlaymiz (pastda)
    # ── "yo'q/нет" → suhbatni offtopic'da qoldiramiz (boshqa nima kerak?) ──
    if history and _NEGATE_RE.match(query):
        # Faqat oldingi bot taklif bergan bo'lsa offtopic'ga yo'naltiramiz,
        # aks holda LLM hal qilsin
        if _offer_intent_from_bot(_last_assistant_msg(history)):
            print(f"[Classifier] negate '{query}' → offtopic (taklif rad etildi)")
            return _r("offtopic", lang)

    # ══════════════════════════════════════════════════════════════════
    # CONTEXT-FIRST: salomlashish (deterministik, arzon — mavzuni o'g'irlamaydi)
    # ══════════════════════════════════════════════════════════════════
    # Salomlashish/identity (qisqa) — mavzuni DAVOM ETTIRMAYDI, har doim offtopic.
    if (is_greeting(query) and len(query.strip().split()) <= 3) or is_identity(query):
        print(f"[Classifier] greeting/identity '{query}' → offtopic")
        return _r("offtopic", lang)

    # Qisqa follow-up'lar ("qachon?", "kechikdimi?", "unda?" ...) endi DINAMIK —
    # ularni LLM aktiv mavzu (active_intent) + history asosida hal qiladi (pastda).
    # Alohida regex ro'yxati ishlatilmaydi.

    # ── LLM bilan klassifikatsiya (history + aktiv mavzu bilan) ─────────
    intent = await _llm_classify(query, history=history, active_intent=active_intent)
    if not intent:
        intent = "faq"
        print("[Classifier] LLM intent topilmadi → default: faq")

    # ── Location bo'lsa, micro+city ham ajratamiz ───────────
    form_query = None
    city_hint  = None
    if intent == "location":
        form_query, city_hint = await _llm_location(query)

    print(
        f"[Classifier] intent={intent} | "
        f"form_query={form_query!r} | city_hint={city_hint!r} | "
        f"lang={lang} | history_len={len(history) if history else 0}"
    )
    return _r(intent, lang, form_query=form_query, city_hint=city_hint)


def _r(
    intent:     str,
    lang:       str,
    barcode:    str | None   = None,
    form_query: str | None   = None,
    city_hint:  str | None   = None,
    lat:        float | None = None,
    lng:        float | None = None,
) -> dict:
    return {
        "intent":     intent,
        "lang":       lang,
        "barcode":    barcode,
        "form_query": form_query,
        "city_hint":  city_hint,
        "lat":        lat,
        "lng":        lng,
    }
