"""
UzPost MultiTurnResolver  (v4.3)
=================================
O'zgarishlar (v4.2 → v4.3):
  - FIX: _is_international_destination() — yangi yordamchi funksiya.
         to_id=None bo'lsa ham to_region nomi bo'yicha xalqaro ekanini aniqlaydi.
         Natijada "Amerikaga", "Rossiyaga" kabi so'rovlarda
         msg_ask_service va _handle_service_selection xalqaro rejimda ishlaydi.
  - FIX: msg_ask_service va _handle_service_selection ikkala joyda ham
         _is_international_destination() ishlatiladi (oldin faqat to_id tekshirilardi).
  - Qolgan barcha qismlar v4.2 bilan bir xil.
"""

import re
import json
import asyncio
import os
from typing import Optional
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher

import aiohttp
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

try:
    from .service_synonyms import (
        SERVICES, SERVICE_SYNONYMS,
        match_service, get_service_name, get_service_emoji,
    )
except ImportError:
    from service_synonyms import (
        SERVICES, SERVICE_SYNONYMS,
        match_service, get_service_name, get_service_emoji,
    )

load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
PRICE_API_BASE = "https://new.pochta.uz/api/v1/calculator/order-price/"
GROQ_MODEL     = os.getenv("GROQ_MODEL_PREMIUM", "llama-3.1-8b-instant")


# ══════════════════════════════════════════════════════════════════
# 0. REGION DATA
# ══════════════════════════════════════════════════════════════════

try:
    from region_data import INTERNATIONAL_IDS, DISTRICTS_BY_REGION, REGIONS, REGION_SYNONYMS
except ImportError:
    INTERNATIONAL_IDS   = set()
    DISTRICTS_BY_REGION = {}
    REGIONS             = {}
    REGION_SYNONYMS     = {}


# ══════════════════════════════════════════════════════════════════
# 1. VILOYAT TANIQLASH
# ══════════════════════════════════════════════════════════════════

REGION_CENTER_IDS: dict[int, str] = {
    318: "Buxoro",
    489: "Navoiy",
    490: "Xorazm",
    484: "Qoraqalpog'iston",
    457: "Samarqand",
    261: "Jizzax",
    360: "Jizzax",
    429: "Sirdaryo",
    273: "Toshkent shahar",
    301: "Toshkent viloyati",
    305: "Andijon",
    334: "Farg'ona",
    456: "Namangan",
    390: "Qashqadaryo",
    444: "Surxondaryo"
}


def _is_region_center(rid: int) -> bool:
    if rid in INTERNATIONAL_IDS:
        return False
    return rid in REGION_CENTER_IDS


def _get_region_key(rid: int) -> Optional[str]:
    name = REGION_CENTER_IDS.get(rid)
    if not name:
        return None
    for key in DISTRICTS_BY_REGION:
        if key in name or name.startswith(key) or key.startswith(name.split()[0]):
            return key
    return name


def _get_districts_for_region(rid: int) -> dict[str, int]:
    region_key = _get_region_key(rid)
    if not region_key or region_key not in DISTRICTS_BY_REGION:
        return {}
    result: dict[str, int] = {}
    for d_id in DISTRICTS_BY_REGION[region_key]:
        for rname, r_id in REGIONS.items():
            if r_id == d_id:
                short = rname.replace(" tumani", "").replace(" shahri", "").replace(" Shahri", "")
                result[short] = d_id
                break
    return result


def _is_district_id(rid: int) -> bool:
    if rid in INTERNATIONAL_IDS:
        return False
    if rid in REGION_CENTER_IDS:
        return False
    for district_ids in DISTRICTS_BY_REGION.values():
        if rid in district_ids:
            return True
    return False


# ══════════════════════════════════════════════════════════════════
# 1b. XALQARO YO'NALISH TANIQLASH  ← v4.3 YANGILIK
# ══════════════════════════════════════════════════════════════════

# to_region nomi bo'yicha xalqaro ekanini aniqlash uchun kalit so'zlar.
# REGION_SYNONYMS da to_id aniqlanmagan bo'lsa ham shu ro'yxat orqali
# xalqaro yo'nalish to'g'ri aniqlanadi.
_INTL_NAME_KEYWORDS: frozenset[str] = frozenset({
    # Eng ko'p ishlatiladigan mamlakatlar (o'zbekcha va inglizcha)
    "rossiya", "russia", "rusiya",
    "america", "amerika", "usa", "aqsh",
    "germaniya", "germany", "nemis",
    "xitoy", "china", "kitay",
    "turkiya", "turkey", "turk",
    "yaponiya", "japan", "yapon",
    "hindiston", "india", "hind",
    "fransiya", "france",
    "italiya", "italy", "italyan",
    "ispaniya", "spain", "ispan",
    "buyuk", "britaniya", "england", "angliya",
    "korea", "koreya",
    "arabiston", "saudi", "saudiya",
    "dubai", "uae", "baa",
    "ukraina", "ukraine",
    "ozarbayjon", "azerbaijan",
    "qozogiston", "qozog", "kazahstan", "kazakhstan",
    "tojikiston", "tajikistan",
    "qirgiziston", "kyrgyzstan",
    "turkmaniston", "turkmenistan",
    "belarusiya", "belarus",
    "gruziya", "georgia",
    "armaniston", "armenia",
    "pokiston", "pakistan",
    "eron", "iran",
    "iroq", "iraq",
    "isroil", "israel",
    "misr", "egypt",
    "kanada", "canada",
    "avstraliya", "australia",
    "braziliya", "brazil",
    "meksika", "mexico",
    "argentina", "argentinya",
    "niderlandiya", "gollandiya", "netherlands", "holland",
    "shvetsiya", "sweden",
    "norvegiya", "norway",
    "finlyandiya", "finland",
    "daniya", "denmark",
    "avstriya", "austria",
    "shveytsariya", "switzerland",
    "polsha", "poland",
    "chexiya", "czechia",
    "vengriya", "hungary",
    "ruminiya", "romania",
    "bolgariya", "bulgaria",
    "serbiya", "serbia",
    "xorvatiya", "croatia",
    "sloveniya", "slovenia",
    "slovakiya", "slovakia",
    "gretsiya", "greece",
    "kipr", "cyprus",
    "portugal", "portugaliya",
    "belgiya", "belgium",
    "singapur", "singapore",
    "malayziya", "malaysia",
    "indoneziya", "indonesia",
    "filippin", "philippines",
    "tailand", "thailand",
    "vetnam", "vietnam",
    "bangladesh",
    "nepal",
    "lankа", "lanka",
    "maldiv",
    "bruney", "brunei",
    "myanmar", "myanma",
    "qatar",
    "bahrayn", "bahrain",
    "quvayt", "kuwait",
    "ummon", "oman",
    "iordaniya", "jordan",
    "livan", "lebanon",
    "liviya", "libya",
    "tunis", "tunisia",
    "marokash", "morocco",
    "jazoir", "algeria",
    "keniya", "kenya",
    "nigeriya", "nigeria",
    "janubiy",   # "Janubiy Koreya", "Janubiy Afrika"
    "shimoliy",  # "Shimoliy Koreya"
    "chernogoriya", "montenegro",
    "albaniya", "albania",
    "makedoniya", "macedonia",
    "moldova",
    "litva", "lithuania",
    "latviya", "latvia",
    "estoniya", "estonia",
    "vatikan", "vatican",
    "luxemburg", "lyuksemburg",
    "vanuatu",
    "ruanda", "rwanda",
    "urugvay", "uruguay",
    "venesuela", "venezuela",
    "kolumbiya", "colombia",
    "peru",
    "chile", "chili",
    "paragvay", "paraguay",
    "bolivia", "boliviya",
    "ekvador", "ecuador",
    "panama",
    "kuba", "cuba",
    "gonduras", "honduras",
    "nikaragua", "nicaragua",
    "gvatemala", "guatemala",
    "kosta", "rica",
    "senegal",
    "angola",
    "ghana", "gana",
    "kamerun", "cameroon",
    "niger",
    "mali",
    "angola",
    "sudan",
    "efiopiya", "ethiopia",
    "tanzaniya", "tanzania",
    "uganda",
    "zambiya", "zambia",
    "zimbabve", "zimbabwe",
    "namibiya", "namibia",
    "mozambik", "mozambique",
    "madagaskar", "madagascar",
    "gabon",
    "kongo", "congo",
    "chad",
    "eritreya", "eritrea",
    "liberiya", "liberia",
    "mavritaniya", "mauritania",
    "jibuti", "djibouti",
    "burkina",
    "ruanda",
    "botsvana", "botswana",
    "gambia",
    "tayvan", "taiwan",
    "gonkong", "hongkong",
    "makao", "macao",
    "bruney",
})


def _is_international_destination(
    to_id: Optional[int],
    to_region: Optional[str],
) -> bool:
    """
    to_id INTERNATIONAL_IDS da bo'lsa YOKI
    to_region nomi xalqaro mamlakat nomiga o'xshasa — True qaytaradi.

    Bu funksiya to_id=None bo'lsa ham (masalan LLM nom chiqardi lekin ID aniqlamadi)
    xalqaro yo'nalishni to'g'ri taniydi.
    """
    if to_id and to_id in INTERNATIONAL_IDS:
        return True
    if to_region:
        clean = re.sub(r"['\u2018\u2019\u02bc\u0060\u02be\u02bf]", "", to_region.lower())
        words = clean.split()
        for w in words:
            if w in _INTL_NAME_KEYWORDS:
                return True
    return False


# ══════════════════════════════════════════════════════════════════
# 2. XIZMAT CHEKLOVLARI
# ══════════════════════════════════════════════════════════════════

SERVICE_WEIGHT_LIMITS: dict[int, Optional[float]] = {
    209: 20.0,   # Bir qadam — max 20 kg
    33:  2.0,    # Xat       — max 2 kg
    136: None,   # Posilka   — cheklov yo'q
    135: None,   # Mayda paket — cheklov yo'q
}


def _check_weight_limit(service_id: int, weight: float, lang: str) -> Optional[str]:
    limit = SERVICE_WEIGHT_LIMITS.get(service_id)
    if limit is None:
        return None
    if weight > limit:
        svc_name = get_service_name(service_id)
        return {
            "uz": (
                f"Afsuski, {svc_name} xizmati orqali {weight} kg jo'natib bo'lmaydi — "
                f"bu xizmat uchun ruxsat etilgan maksimal og'irlik {limit} kg, xolos. "
                f"Iltimos, boshqa xizmat turini tanlang."
            ),
            "ru": (
                f"К сожалению, {weight} кг через {svc_name} отправить не получится — "
                f"максимально допустимый вес для этого вида {limit} кг. "
                f"Пожалуйста, выберите другой способ доставки."
            ),
            "en": (
                f"Unfortunately, {weight} kg cannot be sent via {svc_name} — "
                f"the maximum allowed weight for this service is {limit} kg. "
                f"Please choose a different delivery option."
            ),
        }.get(lang, f"Max og'irlik {limit} kg.")
    return None


# ══════════════════════════════════════════════════════════════════
# 3. CONVERSATION STATE
# ══════════════════════════════════════════════════════════════════

@dataclass
class ShippingState:
    from_region:       Optional[str]   = None
    from_id:           Optional[int]   = None
    to_region:         Optional[str]   = None
    to_id:             Optional[int]   = None
    weight:            Optional[float] = None
    service:           Optional[str]   = None
    service_id:        Optional[int]   = None
    service_confirmed: bool            = False
    turn_count:        int             = 0
    drill_down_field:      Optional[str]  = None
    drill_down_candidates: dict           = field(default_factory=dict)
    drill_down_region:     Optional[str]  = None

    def is_location_complete(self) -> bool:
        return (
            self.from_id is not None
            and self.to_id is not None
            and self.weight is not None
        )

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.from_id:
            missing.append("from_region")
        if not self.to_id:
            missing.append("to_region")
        if self.weight is None:
            missing.append("weight")
        return missing

    def merge(self, other: "ShippingState") -> "ShippingState":
        return ShippingState(
            from_region=other.from_region or self.from_region,
            from_id=other.from_id or self.from_id,
            to_region=other.to_region or self.to_region,
            to_id=other.to_id or self.to_id,
            weight=other.weight if other.weight is not None else self.weight,
            service=self.service,
            service_id=self.service_id,
            service_confirmed=self.service_confirmed,
            turn_count=self.turn_count + 1,
            drill_down_field=self.drill_down_field,
            drill_down_candidates=self.drill_down_candidates,
            drill_down_region=self.drill_down_region,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════
# 4. REGION NORMALIZER + MATCHER
# ══════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "viloyati", "viloyat", "shahri", "tumani", "tuman", "shahar",
    "qishloq", "pochta", "bolim", "uzpost", "yaqin", "uyimdan",
    "uyim", "manzil", "mening", "bizning",
    "oblast", "gorod", "rayon", "respublika", "iz", "v", "do",
    "from", "to", "at", "in", "the", "send", "parcel", "post",
}


def normalize_region(text: str) -> str:
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[ʻʼ'`ʽ\u2018\u2019\u02bc\u02be\u02bf]", "'", t)
    suffixes = ["gacha", "lardan", "larning", "ning", "dan", "ga", "da", "ni", "lar"]
    for suf in suffixes:
        if t.endswith(suf) and len(t) > len(suf) + 3:
            t = t[: -len(suf)]
            break
    t = re.sub(r"[^\w\s']", " ", t)
    words = [w for w in t.split() if w not in _STOPWORDS and len(w) >= 2]
    return " ".join(words).strip()


class RegionMatcher:
    def __init__(self, regions: dict, synonyms: dict):
        self._map: dict[str, int] = {}
        self._build(regions, synonyms)

    def _build(self, regions: dict, synonyms: dict):
        for name, rid in regions.items():
            key = normalize_region(name)
            if key:
                self._map[key] = rid
        for syn, rid in synonyms.items():
            key = normalize_region(syn)
            if key:
                self._map[key] = rid
        extra = {}
        for key, rid in self._map.items():
            words = key.split()
            if len(words) >= 2:
                for w in [words[0], words[-1]]:
                    if w not in self._map and len(w) >= 4:
                        extra[w] = rid
        self._map.update(extra)

    def lookup(self, name: str) -> tuple[Optional[int], float]:
        if not name:
            return None, 0.0
        q = normalize_region(name)
        if not q:
            return None, 0.0
        if q in self._map:
            return self._map[q], 1.0
        for token in sorted(q.split(), key=len, reverse=True):
            if len(token) >= 4 and token in self._map:
                return self._map[token], 0.95
        for key, rid in self._map.items():
            if len(key) >= 4 and len(q) >= 4:
                if q in key.split() or key in q.split():
                    return rid, 0.90
        best_score, best_id = 0.0, None
        for key, rid in self._map.items():
            if abs(len(q) - len(key)) > 4:
                continue
            score = SequenceMatcher(None, q, key).ratio()
            if score > best_score and score >= 0.82:
                best_score, best_id = score, rid
        return best_id, best_score


# ══════════════════════════════════════════════════════════════════
# 5. WEIGHT REGEX
# ══════════════════════════════════════════════════════════════════

_WEIGHT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:kg|кг|kilogram|kilo)\b",
    re.I | re.U,
)

def _extract_weight(text: str) -> Optional[float]:
    m = _WEIGHT_RE.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════
# 6. FAST SCAN — kelishik bo'yicha
# ══════════════════════════════════════════════════════════════════

_FROM_SUFFIX_RE = re.compile(r"([\w'`ʻ\-]{3,}?)(?:dan|дан)\b", re.I | re.U)
_TO_SUFFIX_RE   = re.compile(r"([\w'`ʻ\-]{3,}?)(?:gacha|гача|ga|га)\b", re.I | re.U)


def _synonym_lookup(candidate: str, synonyms: dict) -> tuple[Optional[int], float]:
    key = candidate.lower().strip()
    if key in synonyms:
        return synonyms[key], 1.0
    best_score, best_id = 0.0, None
    for syn, rid in synonyms.items():
        if abs(len(key) - len(syn)) > 3:
            continue
        score = SequenceMatcher(None, key, syn).ratio()
        if score > best_score and score >= 0.85:
            best_score, best_id = score, rid
    return best_id, best_score


def _scan_regions(
    text: str, synonyms: dict,
) -> tuple[Optional[str], Optional[int], Optional[str], Optional[int]]:
    t = text.lower()
    from_name = from_id = to_name = to_id = None

    for m in _FROM_SUFFIX_RE.finditer(t):
        candidate = m.group(1)
        rid, score = _synonym_lookup(candidate, synonyms)
        if rid:
            from_name, from_id = candidate, rid
            print(f"  [Scan/from] '{candidate}' → {rid} ({score:.2f})")
            break

    for m in _TO_SUFFIX_RE.finditer(t):
        candidate = m.group(1)
        rid, score = _synonym_lookup(candidate, synonyms)
        if rid:
            to_name, to_id = candidate, rid
            print(f"  [Scan/to]   '{candidate}' → {rid} ({score:.2f})")
            break

    return from_name, from_id, to_name, to_id


# ══════════════════════════════════════════════════════════════════
# 7. LLM EXTRACTOR
# ══════════════════════════════════════════════════════════════════

def _make_system_prompt(lang: str = "uz") -> str:
    examples = (
        '{"from_region":"Buxoro","to_region":"Toshkent","weight":2}\n'
        '{"from_region":null,"to_region":"Toshkent","weight":null}\n'
        '{"from_region":"G\'ijduvon","to_region":"Marg\'ilon","weight":1}\n'
        '{"from_region":"Rossiya","to_region":"Namangan","weight":null}'
    )
    rules = {
        "uz": (
            "UzPost jo'natma ma'lumot ajratuvchisan.\n"
            "JSON qaytargin: from_region, to_region (O'zbek shahar/tuman YOKI mamlakat|null), weight (kg|null).\n"
            "Qoidalar: taxmin yo'q; kelishik olib tashla (dan/ga/gacha); "
            "uyimdan/manzilimdan→null; faqat JSON.\n"
        ),
        "ru": (
            "Ты экстрактор данных UzPost.\n"
            "Верни JSON: from_region, to_region (узб. город/район ИЛИ страна|null), weight (кг|null).\n"
            "Правила: не угадывай; убирай падежи; uyimdan→null; только JSON.\n"
        ),
        "en": (
            "Extract UzPost shipment data.\n"
            "Return JSON: from_region, to_region (Uzbek city/district OR country|null), weight (kg|null).\n"
            "Rules: no guessing; strip suffixes; uyimdan→null; JSON only.\n"
        ),
    }
    return rules.get(lang, rules["uz"]) + examples


class LLMExtractor:
    def __init__(self, synonyms: dict):
        self._synonyms = synonyms
        self._chains: dict = {}

    def _get_chain(self, lang: str):
        if lang not in self._chains:
            llm = ChatGroq(
                model=GROQ_MODEL, temperature=0.0, max_tokens=100, api_key=GROQ_API_KEY,
            )
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=_make_system_prompt(lang)),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}"),
            ])
            self._chains[lang] = prompt | llm
        return self._chains[lang]

    async def extract(self, query: str, history: list, lang: str = "uz") -> dict:
        weight = _extract_weight(query)
        from_name, from_id, to_name, to_id = _scan_regions(query, self._synonyms)

        if from_name or to_name:
            print(f"[FastExtract] from={from_name}({from_id}) to={to_name}({to_id}) w={weight}")
            return {"from_region": from_name, "to_region": to_name, "weight": weight}

        print("[FastExtract] → LLM")
        return await self._llm_extract(query, history, lang, weight)

    async def _llm_extract(
        self, query: str, history: list, lang: str,
        weight_fallback: Optional[float] = None,
    ) -> dict:
        chain = self._get_chain(lang)
        lc_history = []
        for h_msg, a_msg in history[-2:]:
            lc_history.append(HumanMessage(content=h_msg))
            lc_history.append(AIMessage(content=a_msg))
        try:
            response = await chain.ainvoke({"query": query, "history": lc_history})
            raw = response.content.strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start >= 0 and end > start:
                result = json.loads(raw[start:end])
                if result.get("weight") is None and weight_fallback is not None:
                    result["weight"] = weight_fallback
                return result
        except Exception as e:
            print(f"[LLM] Xato: {e}")
        return {"from_region": None, "to_region": None, "weight": weight_fallback}


# ══════════════════════════════════════════════════════════════════
# 8. API CALL
# ══════════════════════════════════════════════════════════════════

def _get_http_session():
    try:
        from shared_resources import get_http
        return get_http()
    except (RuntimeError, ImportError):
        return None


async def fetch_single_price(
    from_id: int, to_id: int, weight: float, service_id: int,
) -> dict:
    if not all([from_id, to_id, weight, service_id]):
        return {"price": None, "currency": "UZS",
                "error": f"Noto'g'ri parametrlar: from={from_id}, to={to_id}, w={weight}"}

    params = {
        "ServiceTypeId":      service_id,
        "FromJurisdictionId": from_id,
        "Weight":             weight,
        "ToJurisdictionId":   to_id,
    }
    print(f"[API] from={from_id} to={to_id} w={weight} svc={service_id}")

    async def _do(session):
        try:
            async with session.get(
                PRICE_API_BASE, params=params,
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status != 200:
                    return {"price": None, "currency": "UZS", "error": f"HTTP {resp.status}"}
                data = await resp.json()
                if isinstance(data, list) and data:
                    first = data[0]
                    if first.get("status") != "success":
                        return {"price": None, "currency": "UZS",
                                "error": first.get("message", "API xatosi")}
                    price_list = first.get("data", {}).get("list", [])
                    if price_list:
                        total = price_list[0].get("price", {}).get("total")
                        if total is not None:
                            return {"price": float(total), "currency": "UZS", "error": None}
                return {"price": None, "currency": "UZS", "error": "Narx topilmadi"}
        except asyncio.TimeoutError:
            return {"price": None, "currency": "UZS", "error": "Timeout (8 soniya)"}
        except Exception as e:
            return {"price": None, "currency": "UZS", "error": str(e)}

    shared = _get_http_session()
    if shared:
        return await _do(shared)
    async with aiohttp.ClientSession() as session:
        return await _do(session)


# ══════════════════════════════════════════════════════════════════
# 9. MESSAGES
# ══════════════════════════════════════════════════════════════════

def _known_fields_line(state: "ShippingState", lang: str) -> str:
    known = []
    if state.from_region and state.from_id:
        known.append({
            "uz": f"jo'natish joyi: {state.from_region}",
            "ru": f"откуда: {state.from_region}",
            "en": f"from: {state.from_region}",
        }.get(lang, state.from_region))
    if state.to_region and state.to_id:
        known.append({
            "uz": f"yetkazish joyi: {state.to_region}",
            "ru": f"куда: {state.to_region}",
            "en": f"to: {state.to_region}",
        }.get(lang, state.to_region))
    if state.weight is not None:
        known.append({
            "uz": f"og'irlik: {state.weight} kg",
            "ru": f"вес: {state.weight} кг",
            "en": f"weight: {state.weight} kg",
        }.get(lang, f"{state.weight} kg"))
    return ", ".join(known)


def msg_drill_down(
    field: str, region_name: str, districts: dict[str, int],
    lang: str, state: "ShippingState",
) -> str:
    known_line = _known_fields_line(state, lang)
    lines = []

    if known_line:
        confirm = {
            "uz": f"Rahmat, ma'lumotlarni qabul qildim: {known_line}.",
            "ru": f"Спасибо, данные приняты: {known_line}.",
            "en": f"Thank you, I've noted: {known_line}.",
        }.get(lang, "")
        if confirm:
            lines += [confirm, ""]

    if field == "from_region":
        q = {
            "uz": (
                f"{region_name} viloyatidan jo'natmoqchisiz. "
                f"Aniqroq manzil uchun qaysi tuman yoki shahardan ekanini yozing, "
                f"shunda narxni to'g'riroq hisoblaymiz."
            ),
            "ru": (
                f"Вы отправляете из {region_name}. "
                f"Чтобы рассчитать точнее, укажите район или город внутри области."
            ),
            "en": (
                f"You're sending from {region_name} region. "
                f"To give you a more accurate price, please specify the district or city."
            ),
        }.get(lang, f"{region_name} — qaysi tuman?")
    else:
        q = {
            "uz": (
                f"Jo'natma {region_name} viloyatiga yetkazilsin. "
                f"Aniqroq narx uchun qaysi tuman yoki shaharga ekanini yozing."
            ),
            "ru": (
                f"Доставка в {region_name}. "
                f"Уточните район или город — это нужно для точного расчёта."
            ),
            "en": (
                f"Delivery to {region_name} region. "
                f"Please specify the district or city for an accurate calculation."
            ),
        }.get(lang, f"{region_name} — qaysi tuman?")

    lines += [q, ""]
    for i, d_name in enumerate(sorted(districts.keys()), 1):
        lines.append(f"  {i}. {d_name}")

    footer = {
        "uz": "\nTuman nomini yoki ro'yxat raqamini yozing:",
        "ru": "\nНапишите название района или его номер из списка:",
        "en": "\nType the district name or its number from the list:",
    }.get(lang, "\nNom yozing:")
    lines.append(footer)
    return "\n".join(lines)


def msg_ask_service(lang: str, state: Optional["ShippingState"] = None) -> str:
    """
    Xizmat tanlash.
    ← v4.3: is_international endi _is_international_destination() orqali aniqlanadi.
    to_id=None bo'lsa ham to_region nomi bo'yicha xalqaro yo'nalish to'g'ri aniqlanadi.
    """
    is_international = _is_international_destination(
        to_id=state.to_id if state else None,
        to_region=state.to_region if state else None,
    )
    lines: list[str] = []

    if state:
        confirm = {
            "uz": (
                f"Zo'r, barcha ma'lumotlar tayyor. "
                f"{state.from_region} dan {state.to_region} ga, {state.weight} kg jo'natma. "
                f"Endi qaysi xizmat orqali yuborishni tanlang."
            ),
            "ru": (
                f"Отлично, все данные получены. "
                f"Отправление из {state.from_region} в {state.to_region}, {state.weight} кг. "
                f"Теперь выберите способ доставки."
            ),
            "en": (
                f"Perfect, all details are ready. "
                f"Shipment from {state.from_region} to {state.to_region}, {state.weight} kg. "
                f"Now please choose your delivery service."
            ),
        }.get(lang, "")
        if confirm:
            lines += [confirm, ""]

    if is_international:
        intl_note = {
            "uz": (
                "Xalqaro jo'natmalarda faqat quyidagi xizmatlar mavjud "
                "(\"Bir qadam\" faqat O'zbekiston ichida ishlaydi):"
            ),
            "ru": (
                "Для международных отправлений доступны следующие виды "
                "(\"Bir qadam\" работает только внутри Узбекистана):"
            ),
            "en": (
                "For international shipments, the following services are available "
                "(\"Bir qadam\" is only for domestic delivery within Uzbekistan):"
            ),
        }.get(lang, "")
        lines += [intl_note, ""]

        services = {
            "uz": [
                ("1. Posilka",     "tovarlar va ruxsat etilgan barcha narsalar"),
                ("2. Xat",         "hujjatlar va yozishmalar, og'irligi 2 kg gacha"),
                ("3. Mayda paket", "kichik va sinmaydigan buyumlar"),
            ],
            "ru": [
                ("1. Посылка",      "товары и другие разрешённые вещи"),
                ("2. Письмо",       "документы и переписка, до 2 кг"),
                ("3. Мелкий пакет", "небольшие предметы"),
            ],
            "en": [
                ("1. Parcel",       "goods and other permitted items"),
                ("2. Letter",       "documents and correspondence, up to 2 kg"),
                ("3. Small packet", "small non-fragile items"),
            ],
        }
    else:
        domestic_note = {
            "uz": "Quyidagi xizmatlardan birini tanlang:",
            "ru": "Выберите один из следующих вариантов:",
            "en": "Please choose one of the following services:",
        }.get(lang, "")
        lines += [domestic_note, ""]

        services = {
            "uz": [
                ("1. Bir qadam",   "butun O'zbekistonga 1 kunda yetkazish, og'irligi 20 kg gacha"),
                ("2. Posilka",     "tovarlar va ruxsat etilgan barcha narsalar"),
                ("3. Xat",         "hujjatlar va yozishmalar, og'irligi 2 kg gacha"),
                ("4. Mayda paket", "kichik va sinmaydigan buyumlar"),
            ],
            "ru": [
                ("1. Bir qadam",    "доставка за 1 день по всему Узбекистану, до 20 кг"),
                ("2. Посылка",      "товары и другие разрешённые вещи"),
                ("3. Письмо",       "документы и переписка, до 2 кг"),
                ("4. Мелкий пакет", "небольшие предметы"),
            ],
            "en": [
                ("1. Bir qadam",    "next-day delivery across Uzbekistan, up to 20 kg"),
                ("2. Parcel",       "goods and other permitted items"),
                ("3. Letter",       "documents and correspondence, up to 2 kg"),
                ("4. Small packet", "small non-fragile items"),
            ],
        }

    items = services.get(lang, services["uz"])
    for name, desc in items:
        lines.append(f"{name} — {desc}")

    footer = {
        "uz": "\nXizmat nomini yoki raqamini yozing:",
        "ru": "\nНапишите название или номер услуги:",
        "en": "\nType the service name or number:",
    }.get(lang, ":")
    lines.append(footer)
    return "\n".join(lines)


def msg_service_not_found(
    lang: str, is_international: bool = False,
    weight_error: Optional[str] = None,
) -> str:
    if weight_error:
        next_step = {
            "uz": "\n\nQuyidagilardan birini tanlashingiz mumkin:",
            "ru": "\n\nВы можете выбрать один из следующих вариантов:",
            "en": "\n\nYou may choose from the following options:",
        }.get(lang, "")
        return weight_error + next_step + "\n\n" + msg_ask_service(lang)

    if is_international:
        return {
            "uz": (
                "Kechirasiz, bu xizmat nomini tushunmadim. "
                "Xalqaro jo'natmalar uchun quyidagilardan birini tanlang:\n"
                "  1. Posilka\n"
                "  2. Xat — 2 kg gacha\n"
                "  3. Mayda paket\n\n"
                "Raqam yoki nom yozing."
            ),
            "ru": (
                "Извините, не распознал название. "
                "Для международных отправлений выберите:\n"
                "  1. Посылка\n"
                "  2. Письмо — до 2 кг\n"
                "  3. Мелкий пакет\n\n"
                "Напишите номер или название."
            ),
            "en": (
                "Sorry, I didn't recognize that service. "
                "For international shipments please choose:\n"
                "  1. Parcel\n"
                "  2. Letter — up to 2 kg\n"
                "  3. Small packet\n\n"
                "Type the number or name."
            ),
        }.get(lang, "Nom yozing.")

    return {
        "uz": (
            "Kechirasiz, bu xizmat nomini tushunmadim. "
            "Quyidagilardan birini tanlang:\n"
            "  1. Bir qadam — 20 kg gacha\n"
            "  2. Posilka\n"
            "  3. Xat — 2 kg gacha\n"
            "  4. Mayda paket\n\n"
            "Raqam yoki nom yozing."
        ),
        "ru": (
            "Извините, не распознал. "
            "Пожалуйста, выберите из списка:\n"
            "  1. Bir qadam — до 20 кг\n"
            "  2. Посылка\n"
            "  3. Письмо — до 2 кг\n"
            "  4. Мелкий пакет\n\n"
            "Напишите номер или название."
        ),
        "en": (
            "Sorry, I didn't catch that. "
            "Please choose from the list:\n"
            "  1. Bir qadam — up to 20 kg\n"
            "  2. Parcel\n"
            "  3. Letter — up to 2 kg\n"
            "  4. Small packet\n\n"
            "Type the number or name."
        ),
    }.get(lang, "Nom yozing.")


def msg_ask_missing(
    state: "ShippingState", missing: list[str], lang: str,
    unresolved_name: Optional[str] = None,
) -> str:
    known_line = _known_fields_line(state, lang)
    lines = []

    if known_line:
        confirm = {
            "uz": f"Yaxshi, quyidagilarni qabul qildim: {known_line}.",
            "ru": f"Хорошо, следующее зафиксировал: {known_line}.",
            "en": f"Got it, I've noted: {known_line}.",
        }.get(lang, "")
        if confirm:
            lines += [confirm, ""]

    if unresolved_name:
        lines.append({
            "uz": (
                f"Afsuski, \"{unresolved_name}\" manzilini bazamizda topa olmadim. "
                f"Iltimos, aniqroq yoki to'liq shahar/tuman nomini yozing "
                f"(masalan: Buxoro, G'ijduvon, Marg'ilon, Kitob)."
            ),
            "ru": (
                f"К сожалению, \"{unresolved_name}\" не найдено в нашей базе. "
                f"Пожалуйста, напишите название города или района точнее "
                f"(например: Бухара, Гиждуван, Маргилан, Китаб)."
            ),
            "en": (
                f"Unfortunately, \"{unresolved_name}\" wasn't found in our database. "
                f"Please write the city or district name more clearly "
                f"(e.g. Bukhara, Gijduvan, Margilan, Kitob)."
            ),
        }.get(lang, f"\"{unresolved_name}\" topilmadi."))
        return "\n".join(lines)

    first = missing[0]
    qs = {
        "uz": {
            "from_region": (
                "Jo'natmani qayerdan yubormoqchisiz? "
                "Shahar yoki tuman nomini yozing (masalan: Buxoro, G'ijduvon, Kitob)."
            ),
            "to_region": (
                "Jo'natma qayerga yetkazilsin? "
                "Shahar yoki tuman nomini yozing (masalan: Toshkent, Marg'ilon, Nukus)."
            ),
            "weight": (
                "Jo'natmangiz taxminiy og'irligi qancha? "
                "Kilogrammda yozing (masalan: 1.5)."
            ),
        },
        "ru": {
            "from_region": (
                "Откуда будете отправлять? "
                "Напишите город или район (например: Бухара, Гиждуван, Китаб)."
            ),
            "to_region": (
                "Куда нужно доставить? "
                "Напишите город или район (например: Ташкент, Маргилан, Нукус)."
            ),
            "weight": (
                "Какой примерный вес отправления? "
                "Укажите в килограммах (например: 1.5)."
            ),
        },
        "en": {
            "from_region": (
                "Where are you sending from? "
                "Please write the city or district (e.g. Bukhara, Gijduvan, Kitob)."
            ),
            "to_region": (
                "Where should the shipment be delivered? "
                "Please write the city or district (e.g. Tashkent, Margilan, Nukus)."
            ),
            "weight": (
                "What is the approximate weight of your shipment? "
                "Please write it in kilograms (e.g. 1.5)."
            ),
        },
    }
    lines.append(qs.get(lang, qs["uz"]).get(first, first))
    return "\n".join(lines)


def msg_intl_from_blocked(lang: str) -> str:
    return {
        "uz": (
            "Kechirasiz, bizning kalkulyator faqat O'zbekistondan tashqariga "
            "jo'natiladigan jo'natmalar narxini hisoblay oladi. "
            "Chet eldan O'zbekistonga kelayotgan jo'natmalar bu xizmatga kirmaydi.\n\n"
            "Boshqa savolingiz bo'lsa, yordam berishga tayyorman."
        ),
        "ru": (
            "Извините, наш калькулятор рассчитывает стоимость только для отправлений "
            "из Узбекистана за рубеж. "
            "Входящие международные посылки в этот сервис не входят.\n\n"
            "Если есть другой вопрос — с удовольствием помогу."
        ),
        "en": (
            "Sorry, our calculator is designed only for shipments going "
            "out of Uzbekistan. "
            "Incoming international parcels are not covered by this service.\n\n"
            "If you have another question, I'm happy to help."
        ),
    }.get(lang, "Faqat O'zbekistondan xorijga jo'natmalar hisoblanadi.")


def msg_price_result(state: "ShippingState", result: dict, lang: str) -> str:
    svc_name  = get_service_name(state.service_id)
    svc_emoji = get_service_emoji(state.service_id)
    route     = f"{state.from_region} — {state.to_region}"
    w_str = {
        "uz": f"{state.weight} kg",
        "ru": f"{state.weight} кг",
        "en": f"{state.weight} kg",
    }.get(lang, f"{state.weight} kg")

    if result.get("error"):
        return {
            "uz": (
                f"Yo'nalish: {route}, og'irlik: {w_str}, xizmat: {svc_name}.\n\n"
                f"Afsuski, narxni hisoblashda muammo yuz berdi va ma'lumot olishga qiynaldim. "
                f"Iltimos, bir ozdan keyin qayta urinib ko'ring yoki "
                f"to'g'ridan 1165 raqamiga qo'ng'iroq qiling — operatorlarimiz yordam beradi."
            ),
            "ru": (
                f"Маршрут: {route}, вес: {w_str}, услуга: {svc_name}.\n\n"
                f"К сожалению, при расчёте стоимости возникла проблема — "
                f"не удалось получить данные. "
                f"Попробуйте повторить через некоторое время или позвоните по номеру 1165 — "
                f"операторы обязательно помогут."
            ),
            "en": (
                f"Route: {route}, weight: {w_str}, service: {svc_name}.\n\n"
                f"Unfortunately, there was an issue retrieving the price information. "
                f"Please try again shortly or call 1165 — "
                f"our operators will be happy to assist you."
            ),
        }.get(lang, "Xatolik yuz berdi. 1165 ga qo'ng'iroq qiling.")

    price_str = f"{result['price']:,.0f} {result['currency']}"

    # Yangi qoida: qo'shimcha savol FAQAT kalkulyator chala qolganda beriladi.
    # Bu yerda narx allaqachon hisoblangan (to'liq) — shuning uchun
    # "Boshqa yo'nalish narxini hisoblaymi?" kabi follow-up qo'shilmaydi.
    next_step = ""

    return {
        "uz": (
            f"Narx hisoblandi!\n\n"
            f"{svc_emoji} Xizmat: {svc_name}\n"
            f"Yo'nalish: {route}\n"
            f"Og'irlik: {w_str}\n"
            f"To'lov: {price_str}"
            f"{next_step}"
        ),
        "ru": (
            f"Стоимость рассчитана!\n\n"
            f"{svc_emoji} Услуга: {svc_name}\n"
            f"Маршрут: {route}\n"
            f"Вес: {w_str}\n"
            f"Сумма: {price_str}"
            f"{next_step}"
        ),
        "en": (
            f"Price calculated!\n\n"
            f"{svc_emoji} Service: {svc_name}\n"
            f"Route: {route}\n"
            f"Weight: {w_str}\n"
            f"Total: {price_str}"
            f"{next_step}"
        ),
    }.get(lang, f"{svc_name}: {price_str}")


# ══════════════════════════════════════════════════════════════════
# 10. MULTI-TURN RESOLVER
# ══════════════════════════════════════════════════════════════════

class MultiTurnResolver:

    def __init__(self, regions: dict, synonyms: dict):
        self._synonyms      = synonyms
        self.region_matcher = RegionMatcher(regions, synonyms)
        self.llm            = LLMExtractor(synonyms)
        self._sessions: dict[str, tuple[ShippingState, list]] = {}

    # ── Session ───────────────────────────────────────────────────

    def _get_session(self, user_key: str) -> tuple[ShippingState, list]:
        if user_key not in self._sessions:
            self._sessions[user_key] = (ShippingState(), [])
        return self._sessions[user_key]

    def _save_session(self, user_key: str, state: ShippingState, history: list):
        self._sessions[user_key] = (state, history)

    def clear_session(self, user_key: str):
        self._sessions.pop(user_key, None)

    # ── Asosiy metod ──────────────────────────────────────────────

    async def resolve(self, query: str, session_id: str, lang: str = "uz") -> dict:
        state, history = self._get_session(session_id)

        if state.drill_down_field and state.drill_down_candidates:
            return await self._handle_drill_down(query, session_id, state, history, lang)

        if state.is_location_complete() and not state.service_confirmed:
            return await self._handle_service_selection(query, session_id, state, history, lang)

        return await self._handle_location_extraction(query, session_id, state, history, lang)

    # ── Drill-down ────────────────────────────────────────────────

    async def _handle_drill_down(
        self, query: str, session_id: str,
        state: ShippingState, history: list, lang: str,
    ) -> dict:
        candidates = state.drill_down_candidates
        field      = state.drill_down_field
        q          = query.strip()

        chosen_id: Optional[int]   = None
        chosen_name: Optional[str] = None

        if q.isdigit():
            idx = int(q)
            sorted_list = sorted(candidates.items())
            if 1 <= idx <= len(sorted_list):
                chosen_name, chosen_id = sorted_list[idx - 1]

        if not chosen_id:
            q_norm = normalize_region(q)
            for d_name, d_id in candidates.items():
                if normalize_region(d_name) == q_norm:
                    chosen_name, chosen_id = d_name, d_id
                    break

        if not chosen_id:
            best_score = 0.0
            q_norm = normalize_region(q)
            for d_name, d_id in candidates.items():
                score = SequenceMatcher(None, q_norm, normalize_region(d_name)).ratio()
                if score > best_score and score >= 0.75:
                    best_score, chosen_name, chosen_id = score, d_name, d_id

        if not chosen_id:
            fid, _ = self._resolve_region(q)
            if fid and fid in candidates.values():
                chosen_id = fid
                chosen_name = q

        if not chosen_id:
            retry = {
                "uz": (
                    "Kechirasiz, kiritilgan nomni tushunmadim. "
                    "Ro'yxatdan tuman nomini yoki tartib raqamini yozing."
                ),
                "ru": (
                    "Извините, не распознал введённое название. "
                    "Напишите название района из списка или его номер."
                ),
                "en": (
                    "Sorry, I didn't recognize that. "
                    "Please write the district name from the list or its number."
                ),
            }.get(lang, "Raqam yoki nom yozing.")
            return {"status": "need_info", "message": retry, "state": state, "price": None}

        short_name = (chosen_name or "").replace(" tumani", "").replace(" shahri", "").replace(" Shahri", "")
        if field == "from_region":
            state.from_id, state.from_region = chosen_id, short_name
        else:
            state.to_id, state.to_region = chosen_id, short_name

        state.drill_down_field      = None
        state.drill_down_candidates = {}
        state.drill_down_region     = None
        print(f"[DrillDown] {field}='{short_name}' (id={chosen_id})")

        history.append((query, f"{field}={short_name}({chosen_id})"))
        self._save_session(session_id, state, history)

        missing = state.missing_fields()
        if missing:
            return {"status": "need_info",
                    "message": msg_ask_missing(state, missing, lang),
                    "state": state, "price": None}
        return {"status": "need_service",
                "message": msg_ask_service(lang, state=state),
                "state": state, "price": None}

    # ── Location extraction ───────────────────────────────────────

    async def _handle_location_extraction(
        self, query: str, session_id: str,
        state: ShippingState, history: list, lang: str,
    ) -> dict:
        extracted = await self.llm.extract(query, history, lang=lang)
        print(f"[Extracted] {extracted}")

        new_state = self._to_state(extracted)
        merged    = state.merge(new_state)

        if merged.from_id and merged.from_id in INTERNATIONAL_IDS:
            self.clear_session(session_id)
            return {"status": "error", "message": msg_intl_from_blocked(lang),
                    "state": merged, "price": None}

        if merged.from_id and _is_region_center(merged.from_id):
            direct_id, direct_score = self._resolve_region(extracted.get("from_region", ""))
            if direct_id and not _is_region_center(direct_id) and direct_id not in INTERNATIONAL_IDS:
                merged.from_id = direct_id
                merged.from_region = extracted.get("from_region", merged.from_region)
                print(f"[RegionBypass/from] direct district id={direct_id}")
            else:
                districts = _get_districts_for_region(merged.from_id)
                if districts:
                    region_name = merged.from_region or REGION_CENTER_IDS.get(merged.from_id, "?")
                    merged.from_id               = None
                    merged.drill_down_field      = "from_region"
                    merged.drill_down_candidates = districts
                    merged.drill_down_region     = region_name
                    history.append((query, f"Viloyat(from):{region_name}"))
                    self._save_session(session_id, merged, history)
                    return {"status": "need_info",
                            "message": msg_drill_down("from_region", region_name, districts, lang, merged),
                            "state": merged, "price": None}

        if (merged.to_id and _is_region_center(merged.to_id)
                and merged.to_id not in INTERNATIONAL_IDS):
            direct_id, direct_score = self._resolve_region(extracted.get("to_region", ""))
            if direct_id and not _is_region_center(direct_id) and direct_id not in INTERNATIONAL_IDS:
                merged.to_id = direct_id
                merged.to_region = extracted.get("to_region", merged.to_region)
                print(f"[RegionBypass/to] direct district id={direct_id}")
            else:
                districts = _get_districts_for_region(merged.to_id)
                if districts:
                    region_name = merged.to_region or REGION_CENTER_IDS.get(merged.to_id, "?")
                    merged.to_id                 = None
                    merged.drill_down_field      = "to_region"
                    merged.drill_down_candidates = districts
                    merged.drill_down_region     = region_name
                    history.append((query, f"Viloyat(to):{region_name}"))
                    self._save_session(session_id, merged, history)
                    return {"status": "need_info",
                            "message": msg_drill_down("to_region", region_name, districts, lang, merged),
                            "state": merged, "price": None}

        unresolved = None
        raw_from = extracted.get("from_region") or ""
        raw_to   = extracted.get("to_region") or ""
        if raw_from and not merged.from_id:
            unresolved = raw_from
            merged.from_region = None
        elif raw_to and not merged.to_id:
            unresolved = raw_to
            merged.to_region = None

        history.append((query, self._state_summary(merged)))
        self._save_session(session_id, merged, history)

        missing = merged.missing_fields()
        if missing:
            return {"status": "need_info",
                    "message": msg_ask_missing(merged, missing, lang, unresolved_name=unresolved),
                    "state": merged, "price": None}

        return {"status": "need_service",
                "message": msg_ask_service(lang, state=merged),
                "state": merged, "price": None}

    # ── Service selection ─────────────────────────────────────────

    async def _handle_service_selection(
        self, query: str, session_id: str,
        state: ShippingState, history: list, lang: str,
    ) -> dict:
        # ← v4.3: to_id yoki to_region nomi bo'yicha xalqaro aniqlanadi
        is_international = _is_international_destination(
            to_id=state.to_id,
            to_region=state.to_region,
        )

        q = query.strip()
        _INTL_MAP  = {"1": 136, "2": 33, "3": 135}
        _LOCAL_MAP = {"1": 209, "2": 136, "3": 33, "4": 135}

        if q in _INTL_MAP and is_international:
            service_id = _INTL_MAP[q]
        elif q in _LOCAL_MAP and not is_international:
            service_id = _LOCAL_MAP[q]
        else:
            service_id = match_service(query)

        if not service_id:
            return {"status": "need_service",
                    "message": msg_service_not_found(lang, is_international),
                    "state": state, "price": None}

        if is_international and service_id == 209:
            return {"status": "need_service",
                    "message": msg_service_not_found(lang, is_international),
                    "state": state, "price": None}

        weight_err = _check_weight_limit(service_id, state.weight or 0, lang)
        if weight_err:
            return {"status": "need_service",
                    "message": msg_service_not_found(lang, is_international, weight_error=weight_err),
                    "state": state, "price": None}

        state.service_id        = service_id
        state.service           = get_service_name(service_id)
        state.service_confirmed = True
        print(f"[Service] '{q}' → {state.service} (id={service_id})")

        price_result = await fetch_single_price(
            from_id=state.from_id, to_id=state.to_id,
            weight=state.weight, service_id=service_id,
        )
        message = msg_price_result(state, price_result, lang)
        self.clear_session(session_id)

        return {"status": "complete", "message": message, "state": state, "price": price_result}

    # ── Region resolve ────────────────────────────────────────────

    def _resolve_region(self, name: str) -> tuple[Optional[int], float]:
        if not name:
            return None, 0.0
        normalized = normalize_region(name)
        if normalized in self._synonyms:
            return self._synonyms[normalized], 1.0
        for token in normalized.split():
            if len(token) >= 4 and token in self._synonyms:
                return self._synonyms[token], 0.95
        return self.region_matcher.lookup(name)

    def _to_state(self, extracted: dict) -> ShippingState:
        s = ShippingState()
        for field_name, id_attr in [("from_region", "from_id"), ("to_region", "to_id")]:
            raw = extracted.get(field_name) or ""
            if raw:
                rid, score = self._resolve_region(raw)
                setattr(s, field_name, raw)
                setattr(s, id_attr, rid)
                print(f"  [Region] {field_name}='{raw}' → id={rid} ({score:.2f})")
        w = extracted.get("weight")
        if w is not None:
            try:
                s.weight = float(str(w).replace(",", "."))
            except (ValueError, TypeError):
                pass
        return s

    def _state_summary(self, state: ShippingState) -> str:
        parts = []
        if state.from_region: parts.append(f"from={state.from_region}")
        if state.to_region:   parts.append(f"to={state.to_region}")
        if state.weight is not None: parts.append(f"w={state.weight}kg")
        return ", ".join(parts) or "—"

    def _is_service_step(self, user_key: str) -> bool:
        state, _ = self._get_session(user_key)
        return state.is_location_complete() and not state.service_confirmed


# ══════════════════════════════════════════════════════════════════
# 11. FASTAPI HELPER
# ══════════════════════════════════════════════════════════════════

async def handle_price_intent(
    query: str, session_id: str, lang: str, resolver: MultiTurnResolver,
) -> tuple[str, bool, Optional[dict]]:
    result = await resolver.resolve(query, session_id, lang)
    return result["message"], result["status"] == "complete", result.get("price")


# ══════════════════════════════════════════════════════════════════
# 12. TEST
# ══════════════════════════════════════════════════════════════════

async def interactive_test():
    resolver   = MultiTurnResolver(REGIONS, REGION_SYNONYMS)
    session_id = "terminal_user"
    print("\n" + "=" * 55)
    print("  UzPost Narx Kalkulyator  (v4.3)")
    print("  Chiqish: exit")
    print("=" * 55 + "\n")
    while True:
        try:
            query = input("Siz: ").strip()
            if not query or query.lower() in ("exit", "quit", "stop"):
                break
            result = await resolver.resolve(query, session_id, lang="uz")
            print(f"\nBot:\n{result['message']}\n")
            if result["status"] in ("complete", "error"):
                resolver.clear_session(session_id)
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            traceback.print_exc()


async def run_tests():
    resolver = MultiTurnResolver(REGIONS, REGION_SYNONYMS)
    tests = [
        {
            "name": "1. Tuman to'g'ridan",
            "turns": [("G'ijduvondan Toshkentga 2 kg", "uz"), ("bir qadam", "uz")],
        },
        {
            "name": "2. Viloyat → drill-down from",
            "turns": [("Buxorodan Toshkentga 2 kg", "uz"), ("G'ijduvon", "uz"), ("posilka", "uz")],
        },
        {
            "name": "3. Viloyat → drill-down to (Navoiy)",
            "turns": [("G'ijduvondan Navoiyga 1 kg", "uz"), ("Karmana", "uz"), ("xat", "uz")],
        },
        {
            "name": "4. Bir qadam 25 kg → og'irlik xato",
            "turns": [("G'ijduvondan Toshkentga 25 kg", "uz"), ("bir qadam", "uz"), ("posilka", "uz")],
        },
        {
            "name": "5. Xat 3 kg → og'irlik xato",
            "turns": [("G'ijduvondan Toshkentga 3 kg", "uz"), ("xat", "uz"), ("2", "uz")],
        },
        {
            "name": "6. Xalqaro — Bir qadam bloklash (to_id aniq)",
            "turns": [("Toshkentdan Rossiyaga 2 kg", "uz"), ("bir qadam", "uz"), ("2", "uz")],
        },
        {
            "name": "7. From xorijiy → bloklash",
            "turns": [("Rossiyadan Toshkentga 2 kg", "uz")],
        },
        {
            "name": "8. Xalqaro — Amerika (to_id None bo'lsa ham xalqaro xizmat)",
            "turns": [("Toshkentdan Amerikaga 2 kg", "uz"), ("bir qadam", "uz"), ("1", "uz")],
        },
    ]
    for test in tests:
        print(f"\n{'═' * 55}\n  {test['name']}\n{'═' * 55}")
        sid = f"t{abs(hash(test['name'])) % 9999}"
        resolver.clear_session(sid)
        for i, (query, lang) in enumerate(test["turns"], 1):
            print(f"\n  [{i}] \"{query}\"")
            result = await resolver.resolve(query, sid, lang)
            print(f"  → {result['status']}")
            for line in result["message"].split("\n")[:7]:
                if line.strip():
                    print(f"     {line}")
            if result["status"] in ("complete", "error"):
                break


if __name__ == "__main__":
    import sys
    asyncio.run(
        run_tests() if len(sys.argv) > 1 and sys.argv[1] == "test"
        else interactive_test()
    )

