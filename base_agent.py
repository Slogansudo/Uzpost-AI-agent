"""
base_agent.py — Redis yo'q, to'liq in-memory
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class AgentContext:
    query: str
    lang: str
    user_key: str
    intent: str
    conv_state: dict | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class AgentResponse:
    answer: str
    intent: str
    needs_more: bool = False
    next_question: str | None = None
    new_conv_state: dict | None = None
    cached: bool = False


class BaseAgent(ABC):

    async def run(self, ctx: AgentContext) -> AgentResponse:
        try:
            if ctx.conv_state and ctx.conv_state.get("pending_intent") == ctx.intent:
                return await self._handle_multiturn(ctx)
            return await self.handle(ctx)
        except Exception as e:
            print(f"[{self.__class__.__name__} xato] {e}")
            return AgentResponse(
                answer=self._error_message(ctx.lang),
                intent=ctx.intent,
            )

    @abstractmethod
    async def handle(self, ctx: AgentContext) -> AgentResponse:
        pass

    async def handle_turn(self, ctx: AgentContext) -> AgentResponse:
        return await self.handle(ctx)

    async def _handle_multiturn(self, ctx: AgentContext) -> AgentResponse:
        state = ctx.conv_state or {}
        ctx.extra["collected"] = state.get("collected", {})
        ctx.extra["turn"] = state.get("turn", 1) + 1
        return await self.handle_turn(ctx)

    def _error_message(self, lang: str) -> str:
        return {
            "uz": "Kechirasiz, texnik xatolik yuz berdi. Qayta urinib ko'ring yoki 1165 ga murojaat qiling.",
            "ru": "Извините, произошла техническая ошибка. Попробуйте снова или позвоните 1165.",
            "en": "Sorry, a technical error occurred. Please try again or call 1165.",
        }.get(lang, "Kechirasiz, xatolik yuz berdi. 1165")