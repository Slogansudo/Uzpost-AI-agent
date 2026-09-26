"""
conversation/context.py
========================
Har request uchun yagona kontekst obyekti.

ASOSIY G'OYA:
  - Bitta source-of-truth: today_history (DB'dan bugungi xabarlar)
  - Sticky language: bir marta aniqlanadi, "ha"/"yo'q"/"Toshkent" lar
    tilni o'zgartirmaydi
  - Aktiv intent va slot'lar in-memory cache'da saqlanadi (slot'lar tez kerak)
  - Bot javobi history'ga yozilishidan oldin hook va link'lar olib tashlanadi
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ════════════════════════════════════════════════════════════════════
# In-memory cache — faqat slot va aktiv intent (history DB'da)
# Production'da bu joyga Redis qo'yiladi (TTL bilan)
# ════════════════════════════════════════════════════════════════════

_CTX_CACHE: dict[str, dict] = {}


# ════════════════════════════════════════════════════════════════════
# ConversationContext
# ════════════════════════════════════════════════════════════════════

@dataclass
class ConversationContext:
    """
    Har request uchun yagona kontekst.

    Maydonlar:
      user_key       : foydalanuvchi identifikatori
      query          : joriy savol
      today_history  : bugungi xabarlar (DB'dan o'qiladi)
      lang           : sticky til ('uz' | 'ru' | 'en')
      current_intent : oxirgi turn'dagi intent (slot saqlash uchun)
      intent_turn    : shu intent ichida nechinchi turn
      slots          : agent slot'lari (barcode, lat/lng, from_id, ...)
    """
    user_key:       str
    query:          str

    today_history:  list[dict] = field(default_factory=list)
    lang:           str = "uz"

    current_intent: Optional[str] = None
    intent_turn:    int = 0
    slots:          dict = field(default_factory=dict)

    # ─── Helper'lar ──────────────────────────────────────────

    def last_n_messages(self, n: int = 6) -> list[dict]:
        """LLM uchun oxirgi N ta xabar (user+assistant). N=6 → 3 turn."""
        return self.today_history[-n:]

    def last_bot_message(self) -> Optional[str]:
        """Eng oxirgi bot javobi (qaysi savol berganini bilish uchun)."""
        for msg in reversed(self.today_history):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return None

    def last_user_message(self) -> Optional[str]:
        """Joriy query'dan oldingi user xabari."""
        users = [m for m in self.today_history if m.get("role") == "user"]
        return users[-1].get("content") if users else None

    def is_short_reply(self) -> bool:
        """Qisqa javobmi? ("ha", "yo'q", "ok", "davom et")."""
        return len(self.query.strip().split()) <= 2

    def has_history(self) -> bool:
        return bool(self.today_history)


# ════════════════════════════════════════════════════════════════════
# Kontekst yig'ish — har request boshida chaqiriladi
# ════════════════════════════════════════════════════════════════════

async def build_context(
    user_key: str,
    query:    str,
    get_history_fn,   # async funksiya: db_models.session_services.get_history
) -> ConversationContext:
    """
    DB'dan bugungi history'ni o'qiydi, cache'dan slot/intent/lang oladi.

    get_history_fn bo'lib uzatiladi (circular import oldini olish uchun):
      from db_models.session_services import get_history
      ctx = await build_context(user_key, query, get_history)
    """
    # 1. Bugungi history DB'dan (eski → yangi tartibda)
    try:
        today_history = await get_history_fn(user_key, limit=20)
    except Exception as e:
        print(f"[Context] history o'qishda xato: {e}")
        today_history = []

    # 2. Cache'dan oldingi til, intent, slot
    cached = _CTX_CACHE.get(user_key, {})
    prev_lang   = cached.get("lang", "uz")
    prev_intent = cached.get("current_intent")
    prev_turn   = cached.get("intent_turn", 0)
    prev_slots  = dict(cached.get("slots", {}))   # copy — referens muammoga yo'l qo'ymaslik

    ctx = ConversationContext(
        user_key       = user_key,
        query          = query.strip(),
        today_history  = today_history,
        lang           = prev_lang,
        current_intent = prev_intent,
        intent_turn    = prev_turn,
        slots          = prev_slots,
    )

    # 3. Til'ni qayta hisoblaymiz (sticky logic ichida)
    from conversation.language import resolve_language
    ctx.lang = resolve_language(ctx)

    return ctx


# ════════════════════════════════════════════════════════════════════
# Cache yangilash — har request oxirida chaqiriladi
# ════════════════════════════════════════════════════════════════════

def update_cache(
    user_key:    str,
    lang:        str,
    intent:      Optional[str],
    slots:       Optional[dict] = None,
    prev_intent: Optional[str] = None,
) -> None:
    """
    Request oxirida slot va aktiv intent'ni saqlaydi.

    Intent o'zgargan bo'lsa, intent_turn=1, aks holda +1.
    """
    if slots is None:
        slots = {}

    if intent and intent == prev_intent:
        # Avvalgi cache'dan turn olamiz
        prev_turn = _CTX_CACHE.get(user_key, {}).get("intent_turn", 0)
        intent_turn = prev_turn + 1
    else:
        intent_turn = 1 if intent else 0

    _CTX_CACHE[user_key] = {
        "lang":           lang,
        "current_intent": intent,
        "intent_turn":    intent_turn,
        "slots":          dict(slots),
    }


def clear_cache(user_key: str) -> None:
    """Foydalanuvchi cache'ini tozalash (yangi kun yoki /reset)."""
    _CTX_CACHE.pop(user_key, None)


# ════════════════════════════════════════════════════════════════════
# Bot javobini history uchun tozalash
# (Hook, link, telefon, "💬" — bu narsalar history'da kerak emas)
# ════════════════════════════════════════════════════════════════════

# "💬 ..." qatorlar va engagement hook'lar
_HOOK_LINE     = re.compile(r"💬[^\n]*", re.UNICODE)
# Telefon + link footer (1165|https://...)
_FOOTER_LINE   = re.compile(r"1165\s*\|\s*https?://\S+", re.I)
# Yandex Navigator / uz.post linklari
_LINK_LINE     = re.compile(r"https?://\S+")
# Ko'p bo'sh qatorlar
_MULTI_NEWLINE = re.compile(r"\n{2,}")


def strip_for_context(text: str, max_len: int = 250) -> str:
    """
    Bot javobini history yozish va LLM'ga uzatishdan oldin tozalash.

    OLIB TASHLANADI:
      - "💬 ..." hook qatorlar (router uchun shovqin)
      - Telefon + link footer
      - URL'lar (history uchun shart emas)
      - Ko'p bo'sh qatorlar

    Maksimum max_len belgi — token tejaymiz.
    """
    if not text:
        return ""

    cleaned = _HOOK_LINE.sub("", text)
    cleaned = _FOOTER_LINE.sub("", cleaned)
    cleaned = _LINK_LINE.sub("", cleaned)
    cleaned = _MULTI_NEWLINE.sub("\n", cleaned)
    cleaned = cleaned.strip()

    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rsplit(" ", 1)[0] + "..."

    return cleaned
