from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentResponse:
    answer:         str
    cached:         bool = False
    needs_more:     bool = False
    new_conv_state: Optional[dict] = None


class PriceAgent:

    def __init__(self):
        self._resolver = None

    def _get_resolver(self):
        if self._resolver is None:
            from calculator.region_data import REGIONS, REGION_SYNONYMS
            from calculator.region_resolver import MultiTurnResolver
            self._resolver = MultiTurnResolver(REGIONS, REGION_SYNONYMS)
            print("[PriceAgent] MultiTurnResolver tayyor ✓")
        return self._resolver

    def _is_service_step(self, user_key: str) -> bool:
        """
        Resolver xizmat tanlash bosqichidami?

        ESLATMA: Bu metod faqat main.py tomonidan chaqiriladi —
        agent.run() ichida ISHLATILMAYDI (ikki marta chaqirishdan saqlanish uchun).
        agent.run() natijani resolver.resolve() dan kelgan status orqali aniqlaydi.
        """
        if not self._resolver:
            return False
        state, _ = self._resolver._get_session(user_key)
        return state.is_location_complete() and not state.service_confirmed

    async def run(self, user_key: str, query: str, lang: str, turn: int = 1) -> AgentResponse:
        resolver = self._get_resolver()

        # FIX 1: in_service_step ni oldindan CHAQIRMAYMIZ —
        # resolver.resolve() dan kelgan natija orqali aniqlaymiz.
        # Bu _get_session() ni ikki marta chaqirishni oldini oladi.

        # FIX 2: lang ni resolver.resolve() ga uzatamiz
        result  = await resolver.resolve(query=query, session_id=user_key, lang=lang)
        status  = result["status"]
        message = result["message"]
        state   = result.get("state")

        if status in ("complete", "error"):
            return AgentResponse(
                answer=message, cached=False,
                needs_more=False, new_conv_state=None
            )

        # in_service_step ni resolver qaytargan state dan olamiz (bir marta)
        in_service_step = (
            state is not None
            and state.is_location_complete()
            and not state.service_confirmed
        )

        # state_empty: FAQAT hech narsa yig'ilmagan holda True
        # BUG FIX: avval (not from_id AND not to_id AND not weight) edi —
        # bu noto'g'ri: foydalanuvchi bitta ma'lumot bersa (masalan to_id=Toshkent)
        # lekin from_id hali yo'q bo'lsa, state_empty=True deb hisoblar edi.
        # To'g'risi: from_id YOKI to_id YOKI weight bo'lsa — state bo'sh emas.
        state_empty = (
            state is None or
            (not state.from_id and not state.to_id and not state.weight)
        )

        # Xizmat tanlash bosqichida max 4 urinish (3 edi — juda oz)
        # Manzil yig'ishda max 8 urinish (5 edi — G'ijduvon→Toshkent→1.5kg 3 ta savol)
        max_turns = 4 if in_service_step else 8

        if state_empty and turn >= 3:
            # 3 ta so'rovdan keyin ham hech narsa yig'ilmadi — boshqa niyat
            resolver.clear_session(user_key)
            return AgentResponse(
                answer=message, cached=False,
                needs_more=False, new_conv_state=None
            )

        if turn >= max_turns:
            # Turn limit — sessiyani tozala, lekin yaxshi xabar ber
            resolver.clear_session(user_key)
            timeout_msg = {
                "uz": (
                    "Narx hisoblash uchun jo'natish joyi, yetkazish joyi va og'irlikni "
                    "bitta xabarda yozing.\n"
                    "Masalan: \"Toshkentdan Buxoroga 2kg\""
                ),
                "ru": (
                    "Для расчёта укажите в одном сообщении: откуда, куда и вес.\n"
                    "Например: \"Из Ташкента в Бухару, 2 кг\""
                ),
                "en": (
                    "Please write the origin, destination and weight in one message.\n"
                    "Example: \"From Tashkent to Bukhara, 2kg\""
                ),
            }.get(lang, "Qaytadan yozing.")
            return AgentResponse(
                answer=timeout_msg, cached=False,
                needs_more=False, new_conv_state=None
            )

        return AgentResponse(
            answer=message, cached=False,
            needs_more=True,
            new_conv_state={
                "intent":          "price",
                "turn":            turn + 1,
                "lang":            lang,       # FIX 3: lang ni saqlaymiz (resume uchun)
                "in_service_step": in_service_step,
            }
        )

    def clear(self, user_key: str):
        if self._resolver:
            self._resolver.clear_session(user_key)