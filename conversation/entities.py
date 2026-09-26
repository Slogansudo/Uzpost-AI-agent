"""
conversation/entities.py — Entity memory + follow-up detection
================================================================
SUHBAT BOSHQARUVI QATLAMI (agent biznes logikasiga TEGMAYDI).

Maqsad:
  - Foydalanuvchi bir marta bergan entitylarni (trek-kod, manzil/MFY,
    shahar, koordinata, oxirgi mavzu) eslab qolish.
  - Qisqa follow-up xabarlarni ("qachon?", "qayerda?", "unda?", "yana?")
    aniqlash — ular oldingi mavzuning davomi sifatida talqin qilinadi.

Bu modul faqat ma'lumotni SAQLAYDI va ANIQLAYDI. Routing va javob berish
o'zgarmagan agentlar/handlerlar orqali amalga oshiriladi.

Saqlash: in-memory (TTL bilan). Production'da bu joyga Redis qo'yilishi mumkin.
"""
from __future__ import annotations

import re
import time
from typing import Optional

# ── Mavzu (topic) intentlari — last_intent shu intentlarda yangilanadi ────────
# offtopic / prohibited oxirgi mavzuni O'CHIRMAYDI (salom/hazil mavzuni uzmaydi).
TOPIC_INTENTS = {"tracking", "price", "location", "faq"}

# Eslab qolingan entitylar qancha vaqt yashaydi (paused TTL bilan mos)
_TTL_SECONDS = 60 * 20  # 20 daqiqa

_STORE: dict[str, dict] = {}


# ════════════════════════════════════════════════════════════════════
# TREK-KOD (barcode) — joriy xabar va history'dan
# ════════════════════════════════════════════════════════════════════

_BARCODE_RE = re.compile(r"\b([A-Z]{2,4}\d{7,14}[A-Z]{0,2})\b", re.IGNORECASE)


def _valid_barcode(code: str) -> bool:
    code = code.upper()
    if not (10 <= len(code) <= 22):
        return False
    letters = sum(ch.isalpha() for ch in code)
    digits = sum(ch.isdigit() for ch in code)
    return letters >= 2 and digits >= 7


def barcode_from_text(text: str) -> Optional[str]:
    """Matndan to'g'ri trek-kodni ajratadi (tracking/api.py bilan bir xil qoida)."""
    if not text:
        return None
    for c in _BARCODE_RE.findall(text.upper()):
        if _valid_barcode(c):
            return c
    return None


def barcode_from_history(history: list[dict] | None) -> Optional[str]:
    """Suhbat tarixidagi (eng yangi → eski) user xabarlaridan oxirgi trek-kod."""
    if not history:
        return None
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        bc = barcode_from_text(msg.get("content") or "")
        if bc:
            return bc
    return None


# ════════════════════════════════════════════════════════════════════
# QISQA FOLLOW-UP ANIQLASH (DINAMIK — qattiq so'zlar ro'yxatiga bog'liq emas)
# ════════════════════════════════════════════════════════════════════
# G'oya: follow-up'ni aniq so'z bo'yicha emas, STRUKTURA bo'yicha aniqlaymiz:
#   qisqa  +  savol/davom kuesi (?, -mi, -chi, qa-, nima/nega/unda/yana, when/where...)
#   +  yangi mavzu so'zi YO'Q.
# Shunda ro'yxatda bo'lmagan iboralar ham ("kechikdimi?", "yana qancha kutaman?",
# "shunaqami?", "rostmi?") avtomatik follow-up sifatida tushuniladi.

# Follow-up uchun ruxsat etilgan maksimal uzunlik (ChatGPT-vari qisqa davomlar)
_MAX_FOLLOWUP_TOKENS = 5

# ── Savol / davom KUESI (morfologik + keng) ──────────────────────────
# Aniq so'z emas, NAQSH: savol belgisi, -mi/-chi/-ku yuklamalari, qa-* so'zlari,
# umumiy so'roq/davom ravishlari (uz / ru / en).
_QUESTION_CUE_RE = re.compile(
    r"("
    r"\?"                                            # savol belgisi
    r"|\bqa\w*"                                       # qachon, qayer, qaysi, qanaqa, qancha
    r"|\bnec?h\w*|\bnima\w*|\bnega\b|\bkim\b"         # necha, nima, nega, kim
    r"|\w+mi\b|\w+chi\b|\w+ku\b|\w+chi\?"             # -mi/-chi/-ku (keldimi, unchi, borku)
    r"|\bunda\b|\bkeyin\w*|\byana\b|\bhali\b|\btag'?in\b|\bboshqa\b"
    # ── Ruscha ──
    r"|когда|где|куда|скольк\w*|почему|зачем|\bкак\b|каков\w*|\bчто\b|\bкто\b|потом|дальше|ещ[ёе]|\bну\b"
    # ── Inglizcha ──
    r"|\b(when|where|how|why|what|who|then|next|more|else)\b"
    r")",
    re.I | re.U,
)

# ── Yangi mavzu signallari — bular bo'lsa follow-up EMAS (mavzu o'g'irlanmasin) ──
# Qisqa xabarda biror domen so'zi bo'lsa, demak bu mustaqil yangi savol.
_NEW_TOPIC_RE = re.compile(
    r"("
    r"\d+\s*(kg|kilo|kilogramm|gramm|g)\b"            # 5kg, 2 kilo, 500 gramm
    r"|\b("
    r"pochta|posilka|jo'?nat|trek|kuzat|manzil|bo'?lim|filial|"
    r"narx|tarif|to'?lov|kg|kilo|gramm|"
    r"ems|express|ekspres|bir\s*qadam|mayda\s*paket|qadoq|pasport|bojxon|tamojn|"
    r"посылк|отправлен|трек|отделени|адрес|тариф|стоимост|таможн|паспорт|"
    r"parcel|shipment|track|office|branch|price|tariff|customs|passport|packag"
    r")\w*"
    r")",
    re.I | re.U,
)


def looks_like_new_topic(query: str) -> bool:
    """Qisqa xabarda yangi mavzu (domen) so'zi bormi?"""
    return bool(_NEW_TOPIC_RE.search(query or ""))


def is_followup(query: str) -> bool:
    """
    Xabar qisqa follow-up (oldingi mavzuning davomi)mi? — DINAMIK aniqlash.

    True bo'ladi, agar:
      1) xabar qisqa bo'lsa (<= _MAX_FOLLOWUP_TOKENS so'z), VA
      2) salomlashish bo'lmasa, VA
      3) yangi mavzu (domen) so'zi bo'lmasa, VA
      4) savol/davom kuesi bo'lsa (?, -mi, -chi, qa-, nima/nega/unda/yana, when/where...).

    Misollar (ro'yxatsiz, struktura orqali):
      "qachon?", "qayerda?", "unda?", "yana?", "qancha?", "nega?",
      "bo'ldimi?", "kechikdimi?", "yana qancha kutaman?", "shunaqami?", "rostmi?"
    """
    if not query:
        return False
    q = query.strip()
    if len(q.split()) > _MAX_FOLLOWUP_TOKENS:
        return False
    if is_greeting(q):
        return False
    if looks_like_new_topic(q):
        return False
    return bool(_QUESTION_CUE_RE.search(q))


# ════════════════════════════════════════════════════════════════════
# SALOMLASHISH / IDENTITY — mavzuni davom ettirmaydi (har doim offtopic)
# ════════════════════════════════════════════════════════════════════
_GREETING_RE = re.compile(
    r"^\s*(salom|assalom\w*|alik\w*|hi|hello|hey|qandaysan|qalaysan|yaxshimisiz|"
    r"kimsan|kim\s*san|isming\w*|sen\s*kimsan|"
    r"привет|здравств\w*|как\s*дела|ты\s*кто|тебя\s*зовут)\b",
    re.I | re.U,
)


def is_greeting(query: str) -> bool:
    return bool(_GREETING_RE.match((query or "").strip()))


# ════════════════════════════════════════════════════════════════════
# RAD ETISH / BEKOR QILISH  (mustahkam — ko'p so'zli javoblarni ham tutadi)
# ════════════════════════════════════════════════════════════════════
# "yo'q shart emas", "hozircha yo'q", "yo'q kerakmas" kabilar ham tutiladi.
_DECLINE_CUE_RE = re.compile(
    r"\b("
    r"yo'?q|yoq|kerakmas|kerak\s*emas|shart\s*emas|hojati\s*yo'?q|bo'?lmaydi|"
    r"qiziqmayman|qiziq\s*emas|qiziq\s*emas|hech\s*qaysi|bas|to'?xtat\w*|bekor\w*|"
    r"нет|не\s*надо|не\s*нужно|незачем|не\s*интересн\w*|отмена|отмени|стоп|хватит|"
    r"no|nope|nah|not\s*interest\w*|cancel|stop"
    r")\b",
    re.I | re.U,
)

_ACCEPT_CUE_RE = re.compile(
    r"^\s*("
    r"ha+|ha'?a|xo'?p|mayli|albatta|bo'?lad[ia]|davom\s*et\w*|kerak|hisobl\w*|to'?g'?ri|"
    r"ok(ay)?|yes|sure|yeah|yep|yup|"
    r"да|давай|конечно|ага|угу|можно|хочу|продолж\w*|нужно|надо"
    r")\b",
    re.I | re.U,
)


def is_clear_decline(query: str) -> bool:
    """
    Aniq rad / bekor javobimi?
    Qisqa + rad so'zi bor + yangi mavzu so'zi YO'Q.
    Shunda "yo'q shart emas" → True, lekin
    "aloqa bo'limi kerak emas, tarif kerak" → False (yangi so'rov).
    """
    if not query:
        return False
    q = query.strip()
    if len(q.split()) > 6:
        return False
    if not _DECLINE_CUE_RE.search(q):
        return False
    if looks_like_new_topic(q):     # ichida yangi so'rov bor → rad emas
        return False
    return True


def is_clear_accept(query: str) -> bool:
    """
    Aniq rozilik javobimi? ("ha", "ha kerak", "ok", "albatta")
    Yangi mavzu so'zi bo'lsa ("tarif kerak") — rozilik emas.
    """
    if not query:
        return False
    q = query.strip()
    if len(q.split()) > 4:
        return False
    if is_clear_decline(q):
        return False
    if looks_like_new_topic(q):
        return False
    return bool(_ACCEPT_CUE_RE.match(q))


# Eski nomlar bilan moslik (boshqa joylarda ishlatilgan bo'lsa)
def is_negative(query: str) -> bool:
    return is_clear_decline(query)


def is_affirmative(query: str) -> bool:
    return is_clear_accept(query)


# ════════════════════════════════════════════════════════════════════
# IDENTITY / "seni kim yaratgan" — har doim offtopic
# ════════════════════════════════════════════════════════════════════
_IDENTITY_RE = re.compile(
    r"(kim\s*yaratgan|kim\s*yasagan|kim\s*yaratdi|kim\s*ishlab|qaysi\s*kompaniya|"
    r"sen\s*kim|sen\s*nima|isming\s*nima|kim\s*san|qanday\s*ishlay|"
    r"кто\s*(тебя\s*)?создал|кто\s*ты|как\s*тебя|"
    r"who\s*(created|made|are)\s*you|what\s*are\s*you)",
    re.I | re.U,
)


def is_identity(query: str) -> bool:
    return bool(_IDENTITY_RE.search((query or "").strip()))


# ════════════════════════════════════════════════════════════════════
# ENTITY MEMORY
# ════════════════════════════════════════════════════════════════════

def _fresh(user_key: str) -> Optional[dict]:
    e = _STORE.get(user_key)
    if not e:
        return None
    if time.time() - e.get("_ts", 0) > _TTL_SECONDS:
        _STORE.pop(user_key, None)
        return None
    return e


def remember_entities(
    user_key: str,
    *,
    intent: Optional[str] = None,
    barcode: Optional[str] = None,
    form_query: Optional[str] = None,
    city_hint: Optional[str] = None,
    lat: Optional[float] = None,
    lng: Optional[float] = None,
) -> None:
    """
    Turn yakunida entitylarni eslab qoladi.
    - last_intent FAQAT mavzu intentlarida yangilanadi (offtopic uzmaydi).
    - None qiymatlar mavjud eslab qolinganni O'CHIRMAYDI (sticky).
    """
    e = dict(_STORE.get(user_key) or {})
    if intent in TOPIC_INTENTS:
        e["last_intent"] = intent
    if barcode:
        e["barcode"] = barcode
    if form_query:
        e["form_query"] = form_query
    if city_hint:
        e["city_hint"] = city_hint
    if lat is not None and lng is not None:
        e["lat"] = lat
        e["lng"] = lng
    e["_ts"] = time.time()
    _STORE[user_key] = e


def recall_entities(user_key: str) -> dict:
    return dict(_fresh(user_key) or {})


def get_last_intent(user_key: str) -> Optional[str]:
    return recall_entities(user_key).get("last_intent")


def get_remembered_barcode(user_key: str) -> Optional[str]:
    return recall_entities(user_key).get("barcode")


def clear_entities(user_key: str) -> None:
    _STORE.pop(user_key, None)


# ════════════════════════════════════════════════════════════════════
# PENDING OFFER — "qo'shimcha savol" berilgani (tarixdan EMAS, alohida belgi)
# strip_for_context hook'ni history'dan o'chiradi, shuning uchun taklifni
# shu yerda eslab qolamiz. Keyingi turnda ha/yo'q shu belgi orqali aniqlanadi.
# ════════════════════════════════════════════════════════════════════

def set_offer(user_key: str, intent: str) -> None:
    e = dict(_STORE.get(user_key) or {})
    e["offer"] = intent
    e["_ts"] = time.time()
    _STORE[user_key] = e


def get_offer(user_key: str) -> Optional[str]:
    return recall_entities(user_key).get("offer")


def clear_offer(user_key: str) -> None:
    e = _STORE.get(user_key)
    if e and "offer" in e:
        e.pop("offer", None)
