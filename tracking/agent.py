"""
tracking/agent.py — v5.0: Multi-turn history support
======================================================
v5.0 O'ZGARISHLARI:
  - ctx.extra["history"] dan history olinadi va LLM ga uzatiladi
  - Barcha logika o'zgarmagan

O'ZGARMAGAN:
  - BARCODE BOR/YO'Q oqimi
  - sub_intent saqlash
  - _MAX_TRACKING_TURNS
  - needs_more / new_conv_state
"""

from base_agent import BaseAgent, AgentContext, AgentResponse
from tracking.api import extract_barcode, fetch_tracking_data
from tracking.rag import get_tracking_context
from tracking.llm import (
    detect_sub_intent,
    generate_tracking_answer,
    generate_no_barcode_answer,
    generate_fallback_answer,
)

_MAX_TRACKING_TURNS = 3


class TrackingAgent(BaseAgent):

    async def handle(self, ctx: AgentContext) -> AgentResponse:
        query      = ctx.query
        lang       = ctx.lang
        barcode    = ctx.extra.get("barcode") or extract_barcode(query)
        conv_state = ctx.conv_state or {}

        # v5.0: history ctx.extra dan olinadi
        history    = ctx.extra.get("history") or []

        saved_sub = conv_state.get("sub_intent") if conv_state.get("intent") == "tracking" else None
        turn      = conv_state.get("turn", 1) if conv_state.get("intent") == "tracking" else 1

        if barcode:
            detected_sub = detect_sub_intent(query)
            sub_intent   = detected_sub if detected_sub != "status" else (saved_sub or "status")

            print(f"[Tracking] Barcode: {barcode} | sub={sub_intent} | turn={turn} | saved_sub={saved_sub}")

            tracking_data = await fetch_tracking_data(barcode)

            # v5.0: history uzatildi
            answer = await generate_tracking_answer(
                query         = query,
                sub_intent    = sub_intent,
                tracking_data = tracking_data,
                lang          = lang,
                history       = history,
            )

            if not answer:
                answer = generate_fallback_answer(tracking_data, sub_intent, lang)

            return AgentResponse(
                answer    = answer,
                intent    = ctx.intent,
                needs_more= False,
                new_conv_state=None,
            )

        # ── BARCODE YO'Q ──────────────────────────────────────────────────────
        sub_intent = detect_sub_intent(query)

        if saved_sub and sub_intent == "status":
            sub_intent = saved_sub

        print(f"[Tracking] Barcode yo'q | sub={sub_intent} | turn={turn} | saving state")

        rag_ctx = await get_tracking_context(query, sub_intent, lang)

        # v5.0: history uzatildi
        answer = await generate_no_barcode_answer(
            query       = query,
            sub_intent  = sub_intent,
            lang        = lang,
            rag_context = rag_ctx,
            turn        = turn,
            history     = history,
        )

        new_turn = turn + 1
        if new_turn > _MAX_TRACKING_TURNS:
            print(f"[Tracking] {_MAX_TRACKING_TURNS} turn tugadi, state tozalanadi")
            return AgentResponse(
                answer    = answer,
                intent    = ctx.intent,
                needs_more= False,
                new_conv_state=None,
            )

        new_conv_state = {
            "intent":     "tracking",
            "sub_intent": sub_intent,
            "turn":       new_turn,
        }

        return AgentResponse(
            answer        = answer,
            intent        = ctx.intent,
            needs_more    = True,
            new_conv_state= new_conv_state,
        )

    async def handle_turn(self, ctx: AgentContext) -> AgentResponse:
        return await self.handle(ctx)
