"""
conversation — Yagona kontekst boshqaruvi.

Komponentlar:
  - ConversationContext : bitta source-of-truth (history, lang, slots, intent)
  - resolve_language    : sticky til detektori (turn'lar bo'yicha barqaror)
  - build_context       : DB'dan history o'qib, context obyektini yig'adi
  - update_cache        : slot'lar va aktiv intent'ni keyingi turn uchun saqlaydi
  - strip_for_context   : bot javobini history'ga yozishdan oldin tozalaydi
"""
from conversation.context import (
    ConversationContext,
    build_context,
    update_cache,
    clear_cache,
    strip_for_context,
)
from conversation.language import resolve_language

__all__ = [
    "ConversationContext",
    "build_context",
    "update_cache",
    "clear_cache",
    "strip_for_context",
    "resolve_language",
]
