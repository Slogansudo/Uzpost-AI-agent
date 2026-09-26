"""
tracking/llm.py — v5.0: Engagement hook + Multi-turn history
=============================================================
v5.0 O'ZGARISHLARI:
  - generate_tracking_answer() ga history parametri qo'shildi
  - generate_no_barcode_answer() ga history parametri qo'shildi
  - History LangChain messages format'da LLM ga uzatiladi (multi-turn)
  - Promptlar va logika O'ZGARMAGAN

O'ZGARMAGAN:
  - Sub_intent detection
  - _HOOK dictionary va _append_hook
  - _build_tracking_block
  - _SYS, _HINT promptlari
  - Fallback va statik javoblar
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

_GROQ_KEY  = os.getenv("GROQ_API_KEY")
_LLM_MODEL = os.getenv("GROQ_MODEL_PREMIUM", "llama-3.3-70b-versatile")

_LLM: ChatGroq | None = None

def _get_llm() -> ChatGroq:
    global _LLM
    if _LLM is None:
        _LLM = ChatGroq(model=_LLM_MODEL, temperature=0.0, max_tokens=300, api_key=_GROQ_KEY)
    return _LLM


# ── Sub-intent ───────────────────────────────────────────────────────────────

_SUB_KW = {
    "customs": ["tamojn","bojxon","таможн","customs","chegara","clearance","пошлин"],
    "lost":    ["yo'qol","topilm","lost","потерял","пропал"],
    "delay":   ["kelmadi","nega","kech","задержка","не пришёл","delayed",
                "kun bo'ldi","uzoq","vaqt o'tdi","kundan beri","haftadan beri",
                "bo'ldi hali","hafta"],
    "location":["qayerda","filial","manzil","где","отделение","where","adres","bo'limda"],
    "eta":     ["qachon","necha kun","когда","how long","muddati","yetib keladi","прибудет"],
    "no_sms":  ["sms","xabar kelm","смс","уведомл","notification","bildirishm","xabar yo'q"],
    "pickup":  ["olib ket","olsa bo'l","boshqa kishi","получить","другой человек","vakilim","kim oladi"],
}

def detect_sub_intent(query: str) -> str:
    q = query.lower()
    for sub, kws in _SUB_KW.items():
        if any(kw in q for kw in kws):
            return sub
    return "status"


# ── Engagement hook ─────────────────────────────────────────────────────────

_HOOK: dict[str, dict[str, str]] = {
    "delay": {
        "uz": "Boshqa jo'natmangiz ham kechikayaptimi?",
        "ru": "Есть ещё одно задержанное отправление?",
        "en": "Do you have another delayed shipment?",
    },
    "customs": {
        "uz": "Bojxona to'lovi yoki kerakli hujjatlar haqida savoliz bormi?",
        "ru": "Вопросы по таможенным сборам или документам?",
        "en": "Questions about customs fees or required documents?",
    },
    "lost": {
        "uz": "Rasmiy yo'qotish arizasi berish haqida ma'lumot kerakmi?",
        "ru": "Нужна информация о подаче заявления на розыск?",
        "en": "Would you like information on filing a lost item claim?",
    },
    "pickup": {
        "uz": "Bo'lim manzili yoki saqlash muddati haqida savol bormi?",
        "ru": "Вопросы по адресу отделения или сроку хранения?",
        "en": "Questions about branch address or storage time?",
    },
    "eta": {
        "uz": "Boshqa jo'natmaning yetib kelish muddatini ham bilmoqchimisiz?",
        "ru": "Хотите проверить срок доставки другого отправления?",
        "en": "Want to check delivery time for another shipment?",
    },
    "no_sms": {
        "uz": "Bildirishnoma sozlamalari haqida boshqa savol bormi?",
        "ru": "Другие вопросы по настройкам уведомлений?",
        "en": "Other questions about notification settings?",
    },
    "location": {
        "uz": "Bo'lim ish vaqti yoki boshqa manzil haqida savol bormi?",
        "ru": "Вопросы по часам работы или другому адресу?",
        "en": "Questions about branch hours or another location?",
    },
    "status": {
        "uz": "Boshqa jo'natma yoki savol bormi?",
        "ru": "Есть другое отправление или вопрос?",
        "en": "Do you have another shipment or question?",
    },
}

def _hook_recently_shown(history: list[dict] | None) -> bool:
    """Oldingi bot javobi allaqachon 💬 hook bilan tugaganmi?"""
    if not history:
        return False
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return "💬" in (msg.get("content") or "")
    return False


def _append_hook(
    answer: str, sub_intent: str, lang: str,
    history: list[dict] | None = None,
) -> str:
    """
    Engagement hook O'CHIRILGAN. Yangi qoida: qo'shimcha savol FAQAT narx
    kalkulyatori chala qolganda beriladi. Tracking javobi to'liq bo'lgani uchun
    bu yerda hech qanday "Boshqa savol bormi?" qo'shilmaydi.
    """
    return answer


# ── History helper (v5.0) ────────────────────────────────────────────────────

def _build_messages_with_history(
    system_content: str,
    user_content: str,
    history: list[dict] | None = None,
) -> list:
    """
    System + history + current user → messages list.
    History oxirgi 4 ta xabar (2 turn) bilan cheklanadi (token tejash).
    """
    messages = [SystemMessage(content=system_content)]

    if history:
        for msg in history[-6:]:
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=content))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=user_content))
    return messages


# ── API data → LLM bloki ────────────────────────────────────────────────────

def _build_tracking_block(data: dict, lang: str) -> str:
    if not data.get("found"):
        return f"BARCODE:{data.get('barcode','?')} TOPILMADI"

    sk = {"uz": "status_uz", "ru": "status_ru", "en": "status_en"}.get(lang, "status_uz")
    lines = [f"BARCODE:{data['barcode']}"]

    fa = data.get("from_addr", "")
    ta = data.get("to_addr", "")
    if fa or ta:
        lines.append(f"{fa}→{ta}")

    for e in (data.get("events") or [])[:4]:
        s = e.get(sk) or e.get("status_uz", "")
        lines.append(f"{e.get('date','')}|{e.get('place','')}|{s}")

    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYS = {
    "uz": (
        "Sen Pochtachi Bek — UzPost rasmiy yordamchi botisan. "
        "Faqat o'zbek tilida. Tracking ma'lumotiga qat'iy asoslan. "
        "Salomlashma. Emoji yo'q. 3-4 gap. "
        "Faqat aniq ma'lumot topilmasa 1165 yoki uz.post ayt."
    ),
    "ru": (
        "Ты Pochtachi Bek — официальный бот-помощник UzPost. "
        "Только русский. Строго по tracking данным. "
        "Без приветствия. Без эмодзи. 3-4 предложения. "
        "1165 или uz.post — только если данных нет."
    ),
    "en": (
        "You are Pochtachi Bek — official UzPost assistant bot. "
        "English only. Strictly use tracking data. "
        "No greeting. No emoji. 3-4 sentences. "
        "Mention 1165 or uz.post only if data is unavailable."
    ),
}

_HINT = {
    "status":  {"uz":"Holat+marshrut. Bo'limda→pasport/ID,30kun. Yetkazildi→tabrikla.",
                "ru":"Статус+маршрут. Отделение→паспорт/ID,30дн. Доставлено→поздравь.",
                "en":"Status+route. Branch→passport/ID,30d. Delivered→congrats."},
    "delay":   {"uz":"Empatiya+joy+sana. Sabab:bojxona/bayram/viloyat.",
                "ru":"Эмпатия+место+дата. Причина:таможня/праздник/регион.",
                "en":"Empathy+location+date. Reason:customs/holiday/region."},
    "customs": {"uz":"Bojxona holati. 1-7ish kuni. Limit:$200,31kg/oy.",
                "ru":"Таможенный статус. 1-7рабдн. Лимит:$200,31кг/мес.",
                "en":"Customs status. 1-7 biz days. Limit:$200,31kg/mo."},
    "eta":     {"uz":"Joy+taxmin: Toshkent1-2, viloyat2-4, Rossiya10-20kun.",
                "ru":"Место+прогноз: Ташкент1-2, регион2-4, Россия10-20дн.",
                "en":"Location+estimate: Tashkent1-2, region2-4, Russia10-20d."},
    "no_sms":  {"uz":"SMS faqat bo'limda. Tranzitda kelmaydi-normal.",
                "ru":"SMS только в отделении. В пути-нормально.",
                "en":"SMS at branch only. No SMS in transit-normal."},
    "pickup":  {"uz":"Boshqa kishi-pasport/ID. 30kun. Bo'lim manzili.",
                "ru":"Другой человек-паспорт/ID. 30дн. Адрес отделения.",
                "en":"Anyone-passport/ID. 30days. Branch address."},
    "lost":    {"uz":"Empatiya+oxirgi joy. 1165|info@pochta.uz|30ish kuni.",
                "ru":"Эмпатия+последнее место. 1165|info@pochta.uz|30рабдн.",
                "en":"Empathy+last location. 1165|info@pochta.uz|30bizdays."},
    "location":{"uz":"Bo'lim nomi+manzil. Pasport/ID. Du-Ju09-17,Sha09-13.",
                "ru":"Название+адрес. Паспорт/ID. Пн-Пт09-17,Сб09-13.",
                "en":"Branch name+address. Passport/ID. Mon-Fri09-17,Sat09-13."},
}


# ── Barcode BOR → LLM + engagement hook ─────────────────────────────────────

async def generate_tracking_answer(
    query:         str,
    sub_intent:    str,
    tracking_data: dict,
    lang:          str,
    rag_context:   str | None = None,
    history:       list[dict] | None = None,   # ← v5.0 yangi
) -> str | None:
    """
    v5.0: history bilan multi-turn javob.
    history None bo'lsa eski xulqi (single-turn) ishlaydi.
    """
    block = _build_tracking_block(tracking_data, lang)
    hint  = (_HINT.get(sub_intent) or _HINT["status"]).get(lang, "")
    rag   = f"\nRAG:{rag_context.strip()}" if rag_context and rag_context.strip() else ""

    user = f"{query}\n---\n{block}{rag}\n---\n{hint}"

    try:
        # v5.0: history qo'shildi
        messages = _build_messages_with_history(
            system_content=_SYS.get(lang, _SYS["uz"]),
            user_content=user,
            history=history,
        )
        r = await _get_llm().ainvoke(messages)
        answer = r.content.strip()

        if tracking_data.get("found"):
            answer = _append_hook(answer, sub_intent, lang, history=history)

        return answer

    except Exception as e:
        print(f"[LLM/Tracking] {e}")
        return None


# ── Barcode YO'Q → RAG → LLM ─────────────────────────────────────────────────

_ASK = {
    "uz": "Trek raqamingizni yuboring.",
    "ru": "Пришлите трек-номер.",
    "en": "Send your tracking number.",
}

async def generate_no_barcode_answer(
    query:       str,
    sub_intent:  str,
    lang:        str,
    rag_context: str | None = None,
    turn:        int        = 1,
    history:     list[dict] | None = None,   # ← v5.0 yangi
) -> str:
    ask = _ASK.get(lang, _ASK["uz"])

    if rag_context and rag_context.strip():
        hint  = (_HINT.get(sub_intent) or _HINT["status"]).get(lang, "")
        again = "[trek yo'q-yana so'ra]" if turn > 1 else ""
        user  = f"{query}\nTREK:YO'Q{again}\n---\n{rag_context.strip()}\n---\n{hint}\n3-4gap,oxiri:'{ask}'"

        try:
            # v5.0: history qo'shildi
            messages = _build_messages_with_history(
                system_content=_SYS.get(lang, _SYS["uz"]),
                user_content=user,
                history=history,
            )
            r = await _get_llm().ainvoke(messages)
            ans = r.content.strip()
            if ask.split()[0].lower() not in ans.lower():
                ans += f"\n\n{ask}"
            return ans
        except Exception as e:
            print(f"[LLM/NoBarcode] {e}")

    return _STATIC.get(sub_intent, {}).get(lang) or _STATIC["status"].get(lang, ask)


# ── Fallback ─────────────────────────────────────────────────────────────────

def generate_fallback_answer(data: dict, sub_intent: str, lang: str) -> str:
    bc = data.get("barcode", "?")
    if not data.get("found"):
        return {
            "uz": f"{bc} topilmadi. Raqamni tekshiring.\n1165|https://uz.post/uz/tracking/",
            "ru": f"{bc} не найден. Проверьте номер.\n1165|https://uz.post/uz/tracking/",
            "en": f"{bc} not found. Check number.\n1165|https://uz.post/uz/tracking/",
        }.get(lang, f"{bc} topilmadi.")

    sk   = {"uz": "status_uz", "ru": "status_ru", "en": "status_en"}.get(lang, "status_uz")
    evs  = data.get("events") or []
    last = evs[0] if evs else {}
    answer = "\n".join(filter(None, [
        bc,
        f"{data.get('from_addr','')}→{data.get('to_addr','')}",
        f"{last.get('date','—')}|{last.get('place','—')}|{last.get(sk,'—')}",
        "https://uz.post/uz/tracking/|1165",
    ]))

    if data.get("found"):
        answer = _append_hook(answer, sub_intent, lang)

    return answer


# ── Statik (LLM+RAG yo'q) ────────────────────────────────────────────────────

_STATIC: dict[str, dict[str, str]] = {
    "status":  {
        "uz": "Trek raqamingizni yuboring.\nhttps://uz.post/uz/tracking/",
        "ru": "Пришлите трек-номер.\nhttps://uz.post/uz/tracking/",
        "en": "Send tracking number.\nhttps://uz.post/uz/tracking/",
    },
    "customs": {
        "uz": "Xalqaro jo'natmalar bojxonadan o'tadi (1-7 ish kuni). Limit: $200, 31 kg/oy.\n\nTrek raqamingizni yuboring.",
        "ru": "Международные отправления проходят таможню (1-7 рабдн). Лимит: $200, 31 кг/мес.\n\nПришлите трек-номер.",
        "en": "International parcels go through customs (1-7 biz days). Limit: $200, 31 kg/mo.\n\nSend tracking number.",
    },
    "delay":   {
        "uz": "Kechikish sabablari: bojxona, bayramlar, uzoq viloyat.\n\nTrek raqamingizni yuboring.",
        "ru": "Причины: таможня, праздники, удалённый регион.\n\nПришлите трек-номер.",
        "en": "Delays: customs, holidays, remote region.\n\nSend tracking number.",
    },
    "eta":     {
        "uz": "Muddatlar: Toshkent 1-2kun, viloyat 2-4kun, Rossiya 10-20kun, Xitoy 14-30kun.\n\nTrek raqamingizni yuboring.",
        "ru": "Сроки: Ташкент 1-2дн, регион 2-4дн, Россия 10-20дн, Китай 14-30дн.\n\nПришлите трек-номер.",
        "en": "Times: Tashkent 1-2d, region 2-4d, Russia 10-20d, China 14-30d.\n\nSend tracking number.",
    },
    "no_sms":  {
        "uz": "SMS faqat bo'limga yetganda keladi — tranzitda kelmaydi.\n\nTrek raqamingizni yuboring.",
        "ru": "SMS только в отделении — в пути нормально.\n\nПришлите трек-номер.",
        "en": "SMS at branch only — no SMS in transit is normal.\n\nSend tracking number.",
    },
    "pickup":  {
        "uz": "Boshqa kishi olishi mumkin — pasport/ID. Saqlash 30 kun.\n\nTrek raqamingizni yuboring.",
        "ru": "Другой человек — паспорт/ID. Хранение 30 дней.\n\nПришлите трек-номер.",
        "en": "Anyone can pick up — passport/ID. Storage 30 days.\n\nSend tracking number.",
    },
    "lost":    {
        "uz": "1165 (24/7) | info@pochta.uz | Izlash: 30 ish kuni.\n\nTrek raqamingizni yuboring.",
        "ru": "1165 (24/7) | info@pochta.uz | Розыск: 30 рабдн.\n\nПришлите трек-номер.",
        "en": "1165 (24/7) | info@pochta.uz | Search: 30 biz days.\n\nSend tracking number.",
    },
    "location":{
        "uz": "Bo'lim manzilini topish uchun trek raqamingizni yuboring.",
        "ru": "Для адреса отделения пришлите трек-номер.",
        "en": "Send tracking number to find branch address.",
    },
}
