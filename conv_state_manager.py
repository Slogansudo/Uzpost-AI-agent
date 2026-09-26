"""
conv_state_manager.py — Universal Multi-Turn Paused State Manager
=================================================================

MUAMMO (avval):
  Foydalanuvchi price multi-turn davomida "tracking" savol bersa →
  _clear_conv_state() → price sessiyasi yo'qoladi.
  Foydalanuvchi qaytsa → yangi sessiya boshlanadi.

YECHIM (endi):
  active  → hozir shu intent ichida, davom etmoqda
  paused  → foydalanuvchi boshqa mavzuga o'tdi, eski state saqlanadi
  closed  → to'liq tugadi (clear)

PAUSED DAVOMIYLIGI:
  MAX_PAUSED_TURNS = 3  →  3 ta boshqa savoldan keyin avtomatik yopiladi

QAYTISH LOGIKASI:
  Paused state bor + yangi intent = paused intent → resume
  Resume bo'lsa → "davom ettiramizmi?" taklif bilan javob qaytaradi

TRIGGER INTENTLAR (paused hosil qiladiganlar):
  tracking, location, faq  →  paused
  offtopic, prohibited     →  paused EMAS (qisqa, ahamiyatsiz)
  price                    →  paused EMAS (o'zi main intent)

ISHLATISH (main.py da):
  from conv_state_manager import ConvStateManager, ResumeResult

  mgr = ConvStateManager()  # singleton, main.py da bir marta

  # Har bir requestda:
  resume = mgr.check_resume(user_key, current_intent)
  if resume.should_resume:
      # paused intent ni davom ettir
      ...

  # Intent o'zgarganda:
  mgr.pause(user_key, reason_intent=current_intent)

  # Tugaganda:
  mgr.close(user_key)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ─── KONSTANTLAR ──────────────────────────────────────────────────────────────

MAX_PAUSED_TURNS = 3        # necha ta boshqa savoldan keyin paused o'chadi
PAUSED_TTL       = 60 * 15  # 15 daqiqa — shundan keyin eskirgan hisoblanadi

# Bu intentlar kelganda → mavjud active state PAUSED bo'ladi
PAUSE_TRIGGER_INTENTS = {"tracking", "location", "faq"}

# Bu intentlar kelganda → paused state O'CHMAYDI, davom etadi
# (offtopic/prohibited qisqa, ular uchun paused saqlanadi ham)
RESUME_TRIGGER_INTENTS = {"price", "tracking", "location", "faq"}


# ─── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class ConvEntry:
    """Bitta foydalanuvchining to'liq holati."""

    # ACTIVE state
    active_intent:    Optional[str]  = None
    active_data:      dict           = field(default_factory=dict)
    active_since:     float          = field(default_factory=time.time)

    # PAUSED state — eski active
    paused_intent:    Optional[str]  = None
    paused_data:      dict           = field(default_factory=dict)
    paused_at:        Optional[float] = None
    paused_turns_left: int           = MAX_PAUSED_TURNS

    # Qaysi intent sababli paused bo'ldi
    interrupted_by:   Optional[str]  = None

    # Resume hint faqat BIR MARTA ko'rsatiladi (har turn takrorlanmasin)
    resume_hint_shown: bool          = False

    def is_active(self) -> bool:
        return self.active_intent is not None

    def is_paused(self) -> bool:
        return (
            self.paused_intent is not None
            and self.paused_turns_left > 0
            and self.paused_at is not None
            and (time.time() - self.paused_at) < PAUSED_TTL
        )

    def to_active_dict(self) -> Optional[dict]:
        """main.py _CONV_STATE formatiga mos dict qaytaradi."""
        if not self.active_intent:
            return None
        return {"intent": self.active_intent, **self.active_data}

    def to_paused_dict(self) -> Optional[dict]:
        """Paused state ni main.py _CONV_STATE formatiga qaytaradi."""
        if not self.paused_intent:
            return None
        return {"intent": self.paused_intent, **self.paused_data}


@dataclass
class ResumeResult:
    """check_resume() natijasi."""
    should_resume:   bool            = False
    paused_intent:   Optional[str]   = None
    paused_data:     dict            = field(default_factory=dict)
    interrupted_by:  Optional[str]   = None
    resume_hint:     Optional[str]   = None   # LLM javob oxiriga qo'shiladigan taklif


# ─── RESUME HINT MESSAGES ─────────────────────────────────────────────────────

_RESUME_HINTS: dict[str, dict[str, str]] = {
    "price": {
        "uz": "Aytgancha, narx hisoblashni yarim qoldirgandingiz — qayerdan qayerga va necha kg ekanini aytsangiz, davom ettiramiz.",
        "ru": "Кстати, вы не завершили расчёт — напишите откуда, куда и вес, продолжим.",
        "en": "By the way, your price calculation is unfinished — tell me the origin, destination and weight to continue.",
    },
    "tracking": {
        "uz": "Trek raqamingiz hali yuborilmadi — kuzatishni davom ettirish uchun trek raqamni yuboring.",
        "ru": "Трек-номер ещё не был отправлен — пришлите его, чтобы продолжить отслеживание.",
        "en": "Your tracking number hasn't been sent yet — please send it to continue.",
    },
    "location": {
        "uz": "Qaysi hududda pochta bo'limi qidiruvni to'xtatgandingiz — davom ettiramizmi?",
        "ru": "В каком районе вы искали отделение — продолжим поиск?",
        "en": "Which area were you searching for a post office — shall we continue?",
    },
    "faq": {
        "uz": "",
        "ru": "",
        "en": "",
    },
}


# ─── MAIN MANAGER ─────────────────────────────────────────────────────────────

class ConvStateManager:
    """
    Universal multi-turn paused state manager.

    main.py da BITTA nusxa yaratiladi:
        from conv_state_manager import ConvStateManager
        _STATE_MGR = ConvStateManager()

    Har bir requestda ishlatiladi:
        resume = _STATE_MGR.check_resume(user_key, detected_intent, lang)
        _STATE_MGR.set_active(user_key, intent, data)
        _STATE_MGR.pause(user_key, reason_intent)
        _STATE_MGR.close(user_key)
    """

    def __init__(self) -> None:
        self._entries: dict[str, ConvEntry] = {}

    # ── Getterlar ──────────────────────────────────────────────────────────────

    def _get(self, user_key: str) -> ConvEntry:
        if user_key not in self._entries:
            self._entries[user_key] = ConvEntry()
        return self._entries[user_key]

    def get_active(self, user_key: str) -> Optional[dict]:
        """Hozirgi active state ni main.py formatida qaytaradi."""
        entry = self._get(user_key)
        if not entry.is_active():
            return None
        return entry.to_active_dict()

    def get_paused(self, user_key: str) -> Optional[dict]:
        """Paused state ni main.py formatida qaytaradi."""
        entry = self._get(user_key)
        if not entry.is_paused():
            return None
        return entry.to_paused_dict()

    # ── Asosiy metodlar ────────────────────────────────────────────────────────

    def set_active(self, user_key: str, intent: str, data: dict) -> None:
        """
        Yangi active state o'rnatadi.
        Avvalgi active state PAUSED bo'lmaydi — bu pause() orqali amalga oshiriladi.
        """
        entry = self._get(user_key)
        entry.active_intent = intent
        entry.active_data   = {k: v for k, v in data.items() if k != "intent"}
        entry.active_since  = time.time()
        print(f"[ConvState] set_active: {intent} | user={user_key[:8]}")

    def pause(self, user_key: str, reason_intent: str) -> None:
        """
        Active state ni PAUSED ga o'tkazadi.
        reason_intent — qaysi intent sababli to'xtatildi.

        Faqat PAUSE_TRIGGER_INTENTS kelganda chaqiriladi.
        """
        entry = self._get(user_key)
        if not entry.is_active():
            return

        # Eski active → paused ga ko'chirish
        entry.paused_intent      = entry.active_intent
        entry.paused_data        = dict(entry.active_data)
        entry.paused_at          = time.time()
        entry.paused_turns_left  = MAX_PAUSED_TURNS
        entry.interrupted_by     = reason_intent
        entry.resume_hint_shown  = False   # yangi pause → hint qayta ko'rsatilishi mumkin

        # Active tozalanadi
        entry.active_intent = None
        entry.active_data   = {}

        print(
            f"[ConvState] pause: {entry.paused_intent} → paused "
            f"(reason: {reason_intent}) | turns_left={MAX_PAUSED_TURNS}"
        )

    def resume(self, user_key: str) -> Optional[dict]:
        """
        Paused state ni qayta active ga o'tkazadi.
        Qaytaradi: active state dict (main.py formatida) yoki None.
        """
        entry = self._get(user_key)
        if not entry.is_paused():
            return None

        # Paused → active ga qaytarish
        entry.active_intent  = entry.paused_intent
        entry.active_data    = dict(entry.paused_data)
        entry.active_since   = time.time()

        # Paused tozalanadi
        entry.paused_intent      = None
        entry.paused_data        = {}
        entry.paused_at          = None
        entry.paused_turns_left  = 0
        entry.interrupted_by     = None

        result = entry.to_active_dict()
        print(f"[ConvState] resume: → {entry.active_intent} | user={user_key[:8]}")
        return result

    def decrement_paused(self, user_key: str) -> None:
        """
        Har bir boshqa intent kelganda chaqiriladi.
        turns_left kamayadi → 0 ga yetsa paused o'chadi.
        """
        entry = self._get(user_key)
        if not entry.is_paused():
            return
        entry.paused_turns_left -= 1
        print(
            f"[ConvState] paused turns_left: {entry.paused_turns_left} "
            f"(intent={entry.paused_intent})"
        )
        if entry.paused_turns_left <= 0:
            print(f"[ConvState] paused expired: {entry.paused_intent}")
            entry.paused_intent      = None
            entry.paused_data        = {}
            entry.paused_at          = None
            entry.interrupted_by     = None

    def close(self, user_key: str) -> None:
        """Active va paused state ikkalasini ham tozalaydi."""
        if user_key in self._entries:
            del self._entries[user_key]
        print(f"[ConvState] close | user={user_key[:8]}")

    def close_active(self, user_key: str) -> None:
        """Faqat active state ni tozalaydi, paused saqlanadi."""
        entry = self._get(user_key)
        entry.active_intent = None
        entry.active_data   = {}

    # ── check_resume — asosiy qaror funksiyasi ─────────────────────────────────

    def check_resume(
        self,
        user_key:        str,
        current_intent:  str,
        lang:            str = "uz",
    ) -> ResumeResult:
        """
        Har bir requestda chaqiriladi.

        Qaror qoidasi:
          1. Paused state yo'q → ResumeResult(should_resume=False)
          2. Paused state bor + current_intent == paused_intent → RESUME
          3. Paused state bor + current_intent boshqa → decrement, davom
          4. Paused state eskirgan (turns=0 yoki TTL) → o'chadi

        Qaytaradi:
          ResumeResult.should_resume = True  → caller paused intentni davom ettiradi
          ResumeResult.resume_hint           → javob oxiriga qo'shiladigan taklif matni
        """
        entry = self._get(user_key)

        if not entry.is_paused():
            return ResumeResult(should_resume=False)

        paused_intent = entry.paused_intent

        # Case 1: Foydalanuvchi to'g'ridan paused intentga qaytdi
        if current_intent == paused_intent:
            print(f"[ConvState] RESUME detected: {paused_intent}")
            return ResumeResult(
                should_resume=True,
                paused_intent=paused_intent,
                paused_data=dict(entry.paused_data),
                interrupted_by=entry.interrupted_by,
                resume_hint=None,   # davom ettirildi, hint shart emas
            )

        # Case 2: Boshqa intent keldi — turns kamayadi, hint qaytariladi
        # MUHIM: hint FAQAT BIR MARTA ko'rsatiladi (birinchi chetga chiqishda).
        # Aks holda u 3 turn davomida har javobga yopishib qoladi.
        if entry.resume_hint_shown:
            return ResumeResult(
                should_resume=False,
                paused_intent=paused_intent,
                paused_data=dict(entry.paused_data),
                interrupted_by=entry.interrupted_by,
                resume_hint=None,   # allaqachon ko'rsatilgan
            )

        hint_msgs = _RESUME_HINTS.get(paused_intent, {})
        hint      = hint_msgs.get(lang, hint_msgs.get("uz", ""))
        if hint:
            entry.resume_hint_shown = True   # endi qayta chiqmaydi

        return ResumeResult(
            should_resume=False,
            paused_intent=paused_intent,
            paused_data=dict(entry.paused_data),
            interrupted_by=entry.interrupted_by,
            resume_hint=hint,
        )

    # ── should_pause — pause qilish kerakmi? ──────────────────────────────────

    def should_pause_active(
        self,
        user_key:       str,
        incoming_intent: str,
    ) -> bool:
        """
        Active state bor + yangi intent keldi →
        pause qilish kerakmi?

        True bo'lishi uchun:
          - Active state mavjud bo'lishi kerak
          - Kelayotgan intent PAUSE_TRIGGER_INTENTS ichida bo'lishi kerak
          - Kelayotgan intent active intentdan farqli bo'lishi kerak
        """
        entry = self._get(user_key)
        if not entry.is_active():
            return False
        if incoming_intent not in PAUSE_TRIGGER_INTENTS:
            return False
        if incoming_intent == entry.active_intent:
            return False
        return True

    # ── Debug ──────────────────────────────────────────────────────────────────

    def debug_state(self, user_key: str) -> dict:
        entry = self._get(user_key)
        return {
            "active_intent":    entry.active_intent,
            "active_data_keys": list(entry.active_data.keys()),
            "paused_intent":    entry.paused_intent,
            "paused_turns_left": entry.paused_turns_left,
            "paused_ttl_left":  (
                round(PAUSED_TTL - (time.time() - entry.paused_at))
                if entry.paused_at else None
            ),
            "interrupted_by":   entry.interrupted_by,
            "is_active":        entry.is_active(),
            "is_paused":        entry.is_paused(),
        }
