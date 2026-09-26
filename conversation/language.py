"""
conversation/language.py
=========================
Sticky language detektor.

MUAMMO (test loglarda ko'rilgan):
  Turn 1: "Привет детка"       → ru
  Turn 2: "salom"              → uz  ❌ til o'zgardi
  Turn 3: "18:00 da boraman"   → ?    ❌ chalkash

YECHIM:
  - Birinchi turn: detect_lang
  - Keyingi turnlar: aniq til signali bo'lsa SWITCH, aks holda eski tilni saqlash
  - Qisqa javoblar ("ha", "yo'q", "Toshkent", raqamlar) HECH QACHON
    tilni o'zgartirmaydi — chunki ular til-neutral
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conversation.context import ConversationContext


# ════════════════════════════════════════════════════════════════════
# ANIQ til signallari (4+ ta belgi yoki maxsus so'z)
# ════════════════════════════════════════════════════════════════════

# Kirill: 4+ ta ketma-ket kirillcha belgi → ru
_STRONG_RU = re.compile(r"[а-яё]{4,}", re.I | re.U)

# Inglizcha signal so'zlar (faqat kirillsiz tekst'da)
_STRONG_EN = re.compile(
    r"\b(hello|hi|help|please|thank|sorry|can you|how to|where|what|why|when|"
    r"track|parcel|shipping|delivery|nearest|office)\b",
    re.I,
)

# O'zbekcha signal so'zlar (lotin)
_STRONG_UZ = re.compile(
    r"\b(salom|rahmat|iltimos|qanday|nima|qaerda|kerak|bo'?ladi|"
    r"jo'?natma|pochta|kuzatish|qancha|bormi|yordam|"
    r"chi|man|miz|siz|dan|ga|da|ning)\b",
    re.I | re.U,
)

# Kirill umuman bormi?
_ANY_CYRILLIC = re.compile(r"[а-яё]", re.I | re.U)


# ════════════════════════════════════════════════════════════════════
# Asosiy funksiya
# ════════════════════════════════════════════════════════════════════

def resolve_language(ctx: "ConversationContext") -> str:
    """
    Tilni sticky logikaga ko'ra aniqlaydi.

    Qoidalar (tartib bilan tekshiriladi):
      1. Birinchi turn → to'g'ridan detect_lang
      2. Kirillcha 4+ belgi bor → ru (kuchli signal, eski tilni bekor qiladi)
      3. Inglizcha signal so'z + kirill yo'q → en
      4. O'zbekcha signal so'z + kirill yo'q → uz
      5. Qisqa javob (≤2 so'z) yoki raqamlar → ESKI tilni saqlash
      6. Uzun savol → detect_lang, lekin past confidence'da eskini saqlash
    """
    # Circular import oldini olish
    from intent_classifier import detect_lang

    query = ctx.query.strip()

    # ── 1. Birinchi turn ────────────────────────────────────
    if not ctx.has_history():
        detected = detect_lang(query)
        return detected if detected in ("uz", "ru", "en") else "uz"

    # ── 2. Kirillcha kuchli signal → ru ─────────────────────
    if _STRONG_RU.search(query):
        return "ru"

    # ── 3. Inglizcha so'zlar + kirill yo'q → en ─────────────
    has_cyrillic = bool(_ANY_CYRILLIC.search(query))
    if _STRONG_EN.search(query) and not has_cyrillic:
        return "en"

    # ── 4. O'zbekcha so'zlar + kirill yo'q → uz ─────────────
    if _STRONG_UZ.search(query) and not has_cyrillic:
        return "uz"

    # ── 5. Qisqa javob, raqamlar, koordinatalar — eski til ──
    if len(query.split()) <= 2:
        return ctx.lang

    # Faqat raqamlar va belgilar (masalan barcode, koordinata)
    if not re.search(r"[a-zа-яё]", query, re.I | re.U):
        return ctx.lang

    # ── 6. Uzun savol — detect_lang, lekin kichik o'zgarishlar saqlanadi ──
    detected = detect_lang(query)
    if detected not in ("uz", "ru", "en"):
        return ctx.lang

    # Past confidence: agar yangi til oldingidan farq qilsa va kirill yo'q
    # bo'lsa, lotin alifbosi default — uz/en chalkashishi bo'lishi mumkin
    if detected != ctx.lang and not has_cyrillic and detected == "en":
        # uz↔en chalkashishi xavfi yuqori (umumiy so'zlar). Konservativ qoldiramiz.
        if not _STRONG_EN.search(query):
            return ctx.lang

    return detected
