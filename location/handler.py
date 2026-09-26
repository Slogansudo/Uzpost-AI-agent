"""
location/handler.py
===================
Tashqaridan keladi:
  - query      : foydalanuvchi asl so'rovi
  - form_query : mikro-manzil (mfy/ko'cha/massiv) — faqat joy nomi
  - city_hint  : shahar/tuman nomi (classifier dan)
  - lang       : 'uz' | 'ru' | 'en'
  - lat, lng   : to'g'ridan koordinata (ixtiyoriy)

Pipeline:
  ┌─ lat/lng bor        → UzPost API → LLM → javob
  ├─ form_query bor     → NER + city_hint → Yandex → lat/lng
  │                     → UzPost API → LLM → javob
  ├─ count/nechta       → map havola (API chaqirilmaydi)
  └─ hech narsa yo'q    → map havola

O'zgarishlar:
  - _is_offtopic olib tashlandi (classifier ga ishoniladi)
  - _postal_to_text yangi API strukturasiga moslashtirildi:
      working_hours, working_hours_2, working_days, weekend_days
      lat+lng dan Yandex Navigator havolasi yaratiladi
  - _CLARIFY_MSG: micro topilmasa foydalanuvchiga savol
  - [YANGI] Engagement hook: bo'lim topilganda javob oxiriga
      query mazmuniga mos taklif qo'shiladi
"""

import os
import re
import aiohttp
from typing import Optional
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
UZPOST_API     = "https://new.pochta.uz/api/v1/maps/ai/bot/post/offices/by-address/"
YANDEX_GEO     = "https://geocode-maps.yandex.ru/1.x/"
UZ_BBOX        = "55.928,37.184~73.055,45.590"
GROQ_MODEL     = os.getenv("GROQ_MODEL_PREMIUM")


# ════════════════════════════════════════════════════
# FAQAT COUNT ANIQLASH
# ════════════════════════════════════════════════════

_COUNT_RE = re.compile(
    r"\b(nechta|qancha|сколько|how\s+many|jami|hammasi)\b",
    re.I,
)

def _is_count(query: str) -> bool:
    return bool(_COUNT_RE.search(query))


# ════════════════════════════════════════════════════
# ENGAGEMENT HOOK
# Bo'lim topilganda faqat — query mazmuniga qarab aniqlanadi
# ════════════════════════════════════════════════════

# Hook sub-tipi: query dan nima so'ralganini aniqlaymiz
_HOOK_KW: dict[str, list[str]] = {
    "hours": [
        "vaqt", "soat", "ishlaydi", "ochiq", "yopiq", "qachon",
        "время", "часы", "работает", "открыт", "закрыт", "когда",
        "hours", "open", "close", "when",
    ],
    "services": [
        "xizmat", "uzum", "ozon", "ems", "bir qadam", "zoodmall",
        "услуг", "service", "accept", "qabul",
    ],
    "route": [
        "qanday boraman", "yo'l", "marshrut", "navigat", "xarita",
        "как доехать", "дорог", "маршрут", "навигат", "карт",
        "how to get", "direction", "map", "navigate",
    ],
    "nearby": [
        "yaqin", "eng yaqin", "atrofida", "yonida",
        "ближайш", "рядом", "вокруг",
        "nearest", "nearby", "close to",
    ],
}

def _detect_hook_type(query: str) -> str:
    """Query mazmunidan hook turini aniqlaydi."""
    q = query.lower()
    for htype, kws in _HOOK_KW.items():
        if any(kw in q for kw in kws):
            return htype
    return "general"

# Hook matnlari — hook_type + lang
_HOOK: dict[str, dict[str, str]] = {
    "hours": {
        "uz": "Boshqa bo'limning ish vaqti yoki manzilini ham topib beraymi?",
        "ru": "Найти часы работы или адрес другого отделения?",
        "en": "Shall I find hours or address of another branch?",
    },
    "services": {
        "uz": "Bu bo'limdagi boshqa xizmatlar yoki jo'natma narxi haqida savolingiz bormi?",
        "ru": "Вопросы по другим услугам или стоимости отправки в этом отделении?",
        "en": "Questions about other services or shipping rates at this branch?",
    },
    "route": {
        "uz": "Yandex Navigator havolasi yuqorida — yo'l topishda boshqa yordam kerakmi?",
        "ru": "Ссылка на Яндекс Навигатор выше — нужна другая помощь с маршрутом?",
        "en": "The Yandex Navigator link is above — need any other help with directions?",
    },
    "nearby": {
        "uz": "Boshqa hududda ham pochta bo'limi topib beraymi?",
        "ru": "Найти отделение в другом районе?",
        "en": "Shall I find a post office in another area?",
    },
    "general": {
        "uz": "Jo'natma narxini hisoblash yoki kuzatishda yordam kerakmi?",
        "ru": "Нужна помощь с расчётом стоимости или отслеживанием?",
        "en": "Need help with shipping cost or tracking?",
    },
}

def _hook_recently_shown(history: Optional[list[dict]]) -> bool:
    """Oldingi bot javobi allaqachon 💬 hook bilan tugaganmi?"""
    if not history:
        return False
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return "💬" in (msg.get("content") or "")
    return False


def _append_hook(
    answer: str, query: str, lang: str,
    history: Optional[list[dict]] = None,
) -> str:
    """
    Engagement hook O'CHIRILGAN. Yangi qoida: qo'shimcha savol FAQAT narx
    kalkulyatori chala qolganda beriladi. Pochta bo'limi topilgan javob to'liq
    bo'lgani uchun bu yerda hech qanday "Boshqa hududda topib beraymi?" qo'shilmaydi.
    """
    return answer


# ════════════════════════════════════════════════════
# NER: QUERY DAN TUMAN/VILOYAT TOPISH
# ════════════════════════════════════════════════════

_VIL_LATIN = {
    "Ташкент":         "Toshkent",
    "Ташкент вил":     "Toshkent viloyati",
    "Фарғона":         "Farg'ona viloyati",
    "Андижон":         "Andijon viloyati",
    "Наманган":        "Namangan viloyati",
    "Самарканд":       "Samarqand viloyati",
    "Бухоро":          "Buxoro viloyati",
    "Қашқадарё":       "Qashqadaryo viloyati",
    "Сурхондарё":      "Surxondaryo viloyati",
    "Хоразм":          "Xorazm viloyati",
    "Навоий":          "Navoiy viloyati",
    "Жиззах":          "Jizzax viloyati",
    "Сирдарё":         "Sirdaryo viloyati",
    "Қорақалпоғистон": "Qoraqalpog'iston",
}

_APOS_RE = re.compile(r"[ʻʼ`ʹʾ'']")

def _clean(text: str) -> str:
    t = _APOS_RE.sub("'", text.lower().strip())
    t = re.sub(r"(?<=\w)'(?=\w)", "\x00", t)
    t = re.sub(r"[?!.,;:\-«»\"\u2018\u2019\u201c\u201d]", " ", t)
    t = t.replace("\x00", "'")
    return re.sub(r"\s+", " ", t).strip()

_UZ_SUFFIX = re.compile(
    r"(dagi|dagilar|lardan|lardagi|larni|larning|ning|gacha|dan|da|ga|ni|lar|"
    r"mfyda|mfysida|mfysi|ko'chasi|ko'chasida|massivi|massivsida)$",
    re.I,
)

def _strip_uz(word: str) -> str:
    s = _UZ_SUFFIX.sub("", word)
    return s if len(s) >= 3 else word

def _ner_from_query(query: str) -> dict:
    try:
        from adress_data import LEXICON
    except ImportError:
        return {}

    cleaned = _clean(query)
    words   = cleaned.split()

    for window in (3, 2, 1):
        if len(words) < window:
            continue
        for i in range(len(words) - window + 1):
            sl       = words[i: i + window]
            phrase   = " ".join(sl)
            stripped = " ".join(sl[:-1] + [_strip_uz(sl[-1])]) if window > 1 else _strip_uz(sl[0])
            for cand in dict.fromkeys([phrase, stripped]):
                k   = cand.strip()
                hit = LEXICON.get(k) or LEXICON.get(k.replace("'", ""))
                if hit:
                    return hit
    return {}

def _build_yandex_query(
    form_query: str,
    ner:        dict,
    city_hint:  str | None = None,
) -> str:
    parts = [form_query.strip()]

    if city_hint:
        parts.append(city_hint)
        ner_from_city = {}
        try:
            from adress_data import LEXICON
            clean_city    = _clean(city_hint)
            ner_from_city = (
                LEXICON.get(clean_city)
                or LEXICON.get(clean_city.replace("'", ""))
                or {}
            )
        except ImportError:
            pass

        viloyat = ner_from_city.get("viloyat") or ner.get("viloyat")
        if viloyat:
            parts.append(_VIL_LATIN.get(viloyat, viloyat))
    else:
        tuman   = ner.get("tuman")
        shahar  = ner.get("shahar")
        viloyat = ner.get("viloyat")
        if tuman:
            parts.append(tuman)
        elif shahar:
            parts.append(shahar)
        if viloyat:
            parts.append(_VIL_LATIN.get(viloyat, viloyat))

    parts.append("O'zbekiston")
    return ", ".join(parts)


# ════════════════════════════════════════════════════
# YANDEX GEOCODE
# ════════════════════════════════════════════════════

async def _yandex_geocode(
    full_query: str,
    lang:       str,
) -> Optional[tuple[float, float, str]]:
    if not YANDEX_API_KEY:
        print("  [Yandex] API key yo'q")
        return None

    lang_map = {"uz": "uz_UZ", "ru": "ru_RU", "en": "en_US"}
    params = {
        "apikey":  YANDEX_API_KEY,
        "geocode": full_query,
        "format":  "json",
        "results": 5,
        "lang":    lang_map.get(lang, "uz_UZ"),
        "bbox":    UZ_BBOX,
        "rspn":    1,
    }
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                YANDEX_GEO, params=params,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                if r.status != 200:
                    print(f"  [Yandex] HTTP {r.status}")
                    return None
                data    = await r.json()
                members = (
                    data.get("response", {})
                        .get("GeoObjectCollection", {})
                        .get("featureMember", [])
                )
                for m in members:
                    geo   = m.get("GeoObject", {})
                    point = geo.get("Point", {}).get("pos", "")
                    if not point:
                        continue
                    lon_s, lat_s = point.split()
                    lat, lon = float(lat_s), float(lon_s)
                    if 37.0 <= lat <= 45.7 and 55.9 <= lon <= 73.2:
                        fmt = (
                            geo.get("metaDataProperty", {})
                               .get("GeocoderMetaData", {})
                               .get("text", full_query)
                        )
                        print(f"  [Yandex] ({lat:.5f}, {lon:.5f}) → {fmt[:70]}")
                        return lat, lon, fmt
    except Exception as e:
        print(f"  [Yandex] Xato: {e}")
    return None


# ════════════════════════════════════════════════════
# UZPOST API
# ════════════════════════════════════════════════════

async def _uzpost_api(lat: float, lon: float) -> Optional[dict]:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                UZPOST_API,
                params={"lat": lat, "lng": lon},
                timeout=aiohttp.ClientTimeout(total=8),
            ) as r:
                if r.status != 200:
                    print(f"  [UzPost API] HTTP {r.status}")
                    return None
                data   = await r.json()
                postal = data.get("result", {}).get("postal_office")
                if postal:
                    print(f"  [UzPost API] ✓ {postal.get('name_uz','?')} ({postal.get('index','?')})")
                else:
                    print("  [UzPost API] Bo'lim topilmadi")
                return postal
    except Exception as e:
        print(f"  [UzPost API] Xato: {e}")
        return None


# ════════════════════════════════════════════════════
# POSTAL → TEXT  (yangi API strukturasi)
# ════════════════════════════════════════════════════

def _make_yandex_nav_link(lat: str | None, lng: str | None) -> str | None:
    try:
        if lat and lng:
            la, ln = float(lat), float(lng)
            return f"https://yandex.uz/navi/?ll={ln},{la}&z=17&pt={ln},{la}"
    except (ValueError, TypeError):
        pass
    return None

def _postal_to_text(postal: dict, lang: str) -> str:
    name_key = {"uz": "name_uz", "ru": "name_ru", "en": "name_eng"}.get(lang, "name_uz")
    name     = postal.get(name_key) or postal.get("name_uz", "")
    idx      = postal.get("index", "")

    region   = postal.get("region", "") or ""
    city     = postal.get("city", "") or ""
    district = postal.get("district", "") or ""
    mfy      = postal.get("mfy", "") or ""
    street   = postal.get("street", "") or ""
    house    = str(postal.get("house", "") or "").rstrip(".0") or ""
    try:
        house = str(int(float(house))) if house else ""
    except ValueError:
        pass

    addr_parts = []
    if district and district != city:
        addr_parts.append(district)
    elif city:
        addr_parts.append(city)
    if mfy:
        addr_parts.append(mfy)
    if street:
        addr_parts.append(street)
    if house:
        addr_parts.append(house)
    if region and region not in addr_parts:
        addr_parts.append(region)
    addr = ", ".join(filter(None, addr_parts))

    work_days  = postal.get("working_days", "Du-Ju")  or "Du-Ju"
    work_hrs   = postal.get("working_hours", "09:00-17:00") or "09:00-17:00"
    work_days2 = postal.get("working_days_2") or ""
    work_hrs2  = postal.get("working_hours_2") or ""
    weekend    = postal.get("weekend_days", "") or ""

    schedule_parts = [f"{work_days}: {work_hrs}"]
    if work_days2 and work_hrs2:
        schedule_parts.append(f"{work_days2}: {work_hrs2}")
    if weekend:
        schedule_parts.append(
            {"uz": f"Dam olish kuni: {weekend}",
             "ru": f"Выходной: {weekend}",
             "en": f"Day off: {weekend}"}.get(lang, f"Dam olish: {weekend}")
        )
    schedule = " | ".join(schedule_parts)

    _SVC = {
        "uzum":        "Uzum",
        "ozon":        "Ozon",
        "EMS":         "EMS",
        "xalq_global": "Xalq Global",
        "zoodmall":    "ZoodMall",
        "one_step":    "Bir Qadam",
        "bir_qadam":   "Bir Qadam",
    }
    _POS = {"bor", "ha", "yes", "true", "1"}
    seen_svcs = set()
    svcs = []
    for k, lbl in _SVC.items():
        if str(postal.get(k, "")).strip().lower() in _POS and lbl not in seen_svcs:
            svcs.append(lbl)
            seen_svcs.add(lbl)

    geo_link = _make_yandex_nav_link(postal.get("lat"), postal.get("lng"))
    if not geo_link:
        geo_link = postal.get("geolocation") or ""

    lines = [
        f"Bo'lim: {name} (Indeks: {idx})",
        f"Manzil: {addr}",
        f"Ish vaqti: {schedule}",
    ]
    if svcs:
        lines.append(f"Xizmatlar: {', '.join(svcs)}")
    if geo_link:
        lines.append(f"Xarita: {geo_link}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════
# LLM JAVOB
# ════════════════════════════════════════════════════

_SYS = {
    "uz": (
        "Sen Pochtachi Bek — UzPost rasmiy yordamchi botisin. FAQAT O'ZBEK TILIDA javob ber.\n"
        "MANZILNI TUSHUNTIRISH QOIDASI (ENG MUHIM):\n"
        "Manzilni oddiy, tushunarli tarzda yoz — xuddi telefonda yo'l ko'rsatgandek:\n"
        "  - Avval shahar/tuman, keyin ko'cha/MFY, keyin bino raqami\n"
        "  - Atrofdagi taniqli joy (bozor, maktab, masjid) ni ham qo'sh agar ma'lumotda bo'lsa\n"
        "  - 'Bundan ... metrda' yoki 'yonida' kabi yo'naltiruvchi so'zlar ishlat\n\n"
        "JAVOB TUZILISHI:\n"
        "1. Bo'lim nomi va indeksini ayt\n"
        "2. Manzilni yuqoridagi qoida bo'yicha tushunarli qilib yoz\n"
        "3. Ish vaqtini ayt (kun va soat)\n"
        "4. Mavjud xizmatlarni sanab o't\n"
        "5. Yandex Navigator havolasini ber\n"
        "6. OXIRIDA FAQAT aniq manzil/vaqt ma'lumoti bo'lmasa: https://uz.post/uz/map | Tel: 1165\n\n"
        "USLUB:\n"
        "- Samimiy, qisqa, aniq\n"
        "- Foydalanuvchi savolini e'tiborga ol — faqat so'ralgan ma'lumotni ayt\n"
        "- Agar manzil noaniq bo'lsa: 'Aniqroq manzil yozsangiz, yaqinroq bo'limni topaman'\n"
        "- Emoji va markdown ishlatma. 6-9 qator."
    ),
    "ru": (
        "Ты Pochtachi Bek — официальный бот-помощник UzPost. ТОЛЬКО РУССКИЙ ЯЗЫК.\n\n"
        "ПРАВИЛО ОБЪЯСНЕНИЯ АДРЕСА (САМОЕ ВАЖНОЕ):\n"
        "Пиши адрес понятно — как будто объясняешь по телефону, как доехать:\n"
        "  - Сначала город/район, затем улица/МФЙ, затем номер дома\n"
        "  - Добавь ориентир (рынок, школа, мечеть), если есть в данных\n"
        "  - Используй слова-указатели: 'рядом с', 'напротив', 'в ... метрах от'\n\n"
        "СТРУКТУРА ОТВЕТА:\n"
        "1. Название и индекс отделения\n"
        "2. Адрес — понятно, по правилу выше\n"
        "3. Режим работы (дни и часы)\n"
        "4. Доступные услуги\n"
        "5. Ссылка на Яндекс Навигатор\n"
        "6. В КОНЦЕ только если нет точных данных: https://uz.post/uz/map | Тел: 1165\n\n"
        "СТИЛЬ:\n"
        "- Дружелюбно, коротко, по делу\n"
        "- Учитывай вопрос — отвечай именно на то, что спросили\n"
        "- Если адрес неточный: 'Уточните адрес — найду ближайшее отделение'\n"
        "- Без эмодзи и markdown. 6-9 строк."
    ),
    "en": (
        "You are Pochtachi Bek — official UzPost assistant bot. ENGLISH ONLY.\n\n"
        "ADDRESS EXPLANATION RULE (MOST IMPORTANT):\n"
        "Write the address clearly — like giving directions over the phone:\n"
        "  - Start with city/district, then street/MFY, then house number\n"
        "  - Add a nearby landmark (market, school, mosque) if available in the data\n"
        "  - Use guiding phrases: 'next to', 'opposite', 'about ... meters from'\n\n"
        "RESPONSE STRUCTURE:\n"
        "1. Branch name and postal index\n"
        "2. Address — written clearly per the rule above\n"
        "3. Working hours (days and times)\n"
        "4. Available services\n"
        "5. Yandex Navigator link\n"
        "6. END: https://uz.post/uz/map | Tel: 1165 — only if exact data is unavailable\n\n"
        "STYLE:\n"
        "- Friendly, brief, to the point\n"
        "- Focus on what the user asked\n"
        "- If location is unclear: 'Share a more specific address and I'll find the nearest branch'\n"
        "- No emoji or markdown. 6-9 lines."
    ),
}


def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.15,
        max_tokens=400,
        api_key=GROQ_API_KEY,
    )

async def _llm(query: str, postal_text: str, lang: str, history: list[dict] | None = None) -> str:
    """v5.0: history bilan multi-turn javob"""
    if not GROQ_API_KEY:
        return postal_text + "\n\nBatafsil: https://uz.post/uz/map | Tel: 1165"

    user_msg = (
        f'Foydalanuvchi so\'rovi: "{query}"\n\n'
        f"Topilgan pochta bo'limi ma'lumotlari:\n{postal_text}"
    )
    try:
        llm      = _get_llm()
        messages = [
            SystemMessage(content=_SYS.get(lang, _SYS["uz"])),
        ]
        # v5.0: history qo'shamiz (oxirgi 4 xabar = 2 turn)
        if history:
            for msg in history[-4:]:
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=content))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=content))

        messages.append(HumanMessage(content=user_msg))
        resp = await llm.ainvoke(messages)
        return resp.content.strip()
    except Exception as e:
        print(f"  [LLM] Xato: {e}")
        return postal_text + "\n\nBatafsil: https://uz.post/uz/map | Tel: 1165"


# ════════════════════════════════════════════════════
# STANDART JAVOBLAR
# ════════════════════════════════════════════════════

def _map_reply(lang: str) -> str:
    return {
        "uz": (
            "Kechirasiz, bu bo'yicha aniq ma'lumotga ega emasman.\n"
            "Aloqa bo'limlari joylashuvi bo'yicha batafsil ma'lumot:\n"
            "https://uz.post/uz/map | Tel: 1165"
        ),
        "ru": (
            "Извините, точной информацией не располагаю.\n"
            "Подробная информация о расположении отделений связи:\n"
            "https://uz.post/uz/map | Тел: 1165"
        ),
        "en": (
            "Sorry, I don't have specific information on this.\n"
            "For detailed branch locations:\n"
            "https://uz.post/uz/map | Tel: 1165"
        ),
    }.get(lang, "https://uz.post/uz/map | Tel: 1165")


def _not_found(lang: str, hint: str = "") -> str:
    h = f" ({hint})" if hint else ""
    return {
        "uz": (
            f"Kechirasiz{h}, bu manzil bo'yicha pochta bo'limi topilmadi.\n"
            "Barcha bo'limlar: https://uz.post/uz/map | Tel: 1165"
        ),
        "ru": (
            f"Извините{h}, почтовое отделение по этому адресу не найдено.\n"
            "Все отделения: https://uz.post/uz/map | Тел: 1165"
        ),
        "en": (
            f"Sorry{h}, no post office found at this location.\n"
            "All branches: https://uz.post/uz/map | Tel: 1165"
        ),
    }.get(lang, f"Not found. https://uz.post/uz/map | Tel: 1165")


_CLARIFY_MSG = {
    "uz": (
        "Ha, bu hududda pochta bo'limi bor.\n"
        "Ko'cha yoki MFY nomini ham aytсangiz, to'liq manzil va ish vaqtini beraman."
    ),
    "ru": (
        "Да, в этом районе есть отделение.\n"
        "Назовите улицу или МФЙ — дам точный адрес и режим работы."
    ),
    "en": (
        "Yes, there's a post office in this area.\n"
        "Tell me the street or MFY name — I'll give you the exact address and hours."
    ),
}

# ════════════════════════════════════════════════════
# ASOSIY ENTRY POINT
# ════════════════════════════════════════════════════

async def handle(
    query:      str,
    form_query: Optional[str]   = None,
    city_hint:  Optional[str]   = None,
    lang:       str             = "uz",
    lat:        Optional[float] = None,
    lng:        Optional[float] = None,
    ask_micro:  bool            = False,
    history:    Optional[list[dict]] = None,   # ← v5.0 yangi
) -> str:
    # ── 0. Count → map havola ─────────────────────────────────────────
    if _is_count(query):
        print("  [Route] → count → map reply")
        return _map_reply(lang)

    # ── 0b. Micro yo'q, city_hint bor → clarify ──────────────────────
    if ask_micro:
        print(f"  [Route] → ask_micro → clarify ({city_hint})")
        return _CLARIFY_MSG.get(lang, _CLARIFY_MSG["uz"])

    # ── 1. Lat/Lng to'g'ridan ─────────────────────────────────────────
    if lat is not None and lng is not None:
        print(f"  [Route] → lat/lng ({lat}, {lng}) → UzPost API")
        postal = await _uzpost_api(lat, lng)
        if not postal:
            return _not_found(lang, f"{lat:.4f},{lng:.4f}")
        answer = await _llm(query, _postal_to_text(postal, lang), lang, history=history)
        return _append_hook(answer, query, lang, history=history)  # ← hook

    # ── 2. Form query → NER + city_hint → Yandex → UzPost API ────────
    if form_query:
        ner      = _ner_from_query(query)
        yandex_q = _build_yandex_query(form_query, ner, city_hint)
        print(f"  [Route] → form_query boyitildi: '{yandex_q}'")

        geo = await _yandex_geocode(yandex_q, lang)

        if not geo and city_hint:
            fallback_q = f"{form_query}, {city_hint}, O'zbekiston"
            print(f"  [Yandex] Fallback (city_hint): '{fallback_q}'")
            geo = await _yandex_geocode(fallback_q, lang)

        if not geo:
            fallback_q = form_query.strip()
            if "o'zbekiston" not in fallback_q.lower() and "uzbekistan" not in fallback_q.lower():
                fallback_q += ", O'zbekiston"
            print(f"  [Yandex] Fallback (raw): '{fallback_q}'")
            geo = await _yandex_geocode(fallback_q, lang)

        if not geo:
            hint = (form_query + (f", {city_hint}" if city_hint else ""))[:50]
            return _not_found(lang, hint)

        geo_lat, geo_lon, geo_fmt = geo
        print(f"  [Route] → UzPost API ({geo_lat:.5f}, {geo_lon:.5f})")

        postal = await _uzpost_api(geo_lat, geo_lon)
        if not postal:
            return _not_found(lang, geo_fmt[:40])

        answer = await _llm(query, _postal_to_text(postal, lang), lang, history=history)
        return _append_hook(answer, query, lang, history=history)  # ← hook

    # ── 3. Faqat city_hint bor (form_query yo'q) ─────────────────────
    if city_hint:
        yandex_q = f"{city_hint}, O'zbekiston"
        print(f"  [Route] → faqat city_hint: '{yandex_q}'")
        geo = await _yandex_geocode(yandex_q, lang)
        if geo:
            geo_lat, geo_lon, _ = geo
            postal = await _uzpost_api(geo_lat, geo_lon)
            if postal:
                answer = await _llm(query, _postal_to_text(postal, lang), lang, history=history)
                return _append_hook(answer, query, lang, history=history)  # ← hook
        return _not_found(lang, city_hint)

    # ── 4. Hech narsa yo'q → map havola ──────────────────────────────
    print("  [Route] → no location info → map reply")
    return _map_reply(lang)
