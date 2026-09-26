"""
UzPost — Xizmat nomlari va sinonimlar
======================================
Foydalanuvchi har xil tilda/yozuvda yozishi mumkin.
Barcha variantlar shu faylda.
"""

# ── Asosiy xizmatlar (kerak emaslari olib tashlandi) ─────────────────────────
SERVICES = [
    {"id": 209, "name": "Bir qadam",   "emoji": "⚡"},
    {"id": 136, "name": "Posilka",     "emoji": "📦"},
    {"id": 33,  "name": "Xat",         "emoji": "✉️"},
    {"id": 135, "name": "Mayda paket", "emoji": "📫"},
]

# ── Synonym map: kalit → service_id ──────────────────────────────────────────
SERVICE_SYNONYMS: dict[str, int] = {

    # ═══════════════════════════════
    #  BIR QADAM (209) — express
    # ═══════════════════════════════
    "bir qadam":     209,
    "birqadam":      209,
    "bir-qadam":     209,
    "bir q":         209,
    "bq":            209,
    "1": 209,
    "2": 210,
    "3": 211,
    "4": 212,
    "1qadam":        209,
    "1 qadam":       209,
    "express":       209,
    "ekspres":       209,
    "экспресс":      209,
    "tez":           209,
    "tezkor":        209,
    "shoshilinch":   209,
    "shoshilinch":   209,
    "срочно":        209,
    "urgent":        209,
    "fast":          209,
    "quick":         209,
    "tez yetkazish": 209,
    "bir q.":        209,

    # ═══════════════════════════════
    #  POSILKA (136) — oddiy jo'natma
    # ═══════════════════════════════
    "posilka":       136,
    "посылка":       136,
    "parcel":        136,
    "packet":        136,
    "paket":         136,
    "пакет":         136,
    "yuk":           136,
    "юк":            136,
    "cargo":         136,
    "jo'natma":      136,
    "жўнатма":       136,
    "shipment":      136,
    "pos":           136,
    "poss":          136,
    "posil":         136,
    "posel":         136,   # noto'g'ri yozuv
    "posilkа":       136,   # kirill a bilan
    "обычная":       136,
    "standard":      136,
    "oddiy":         136,
    "package":       136,

    # ═══════════════════════════════
    #  XAT (33) — hujjat / konvert
    # ═══════════════════════════════
    "xat":           33,
    "хат":           33,
    "letter":        33,
    "письмо":        33,
    "maktub":        33,
    "конверт":       33,
    "envelope":      33,
    "hujjat":        33,
    "документ":      33,
    "document":      33,
    "xat-xabar":     33,
    "mail":          33,
    "pochta xat":    33,

    # ═══════════════════════════════
    #  MAYDA PAKET (135) — kichik buyum
    # ═══════════════════════════════
    "mayda paket":   135,
    "maydapaket":    135,
    "mayda-paket":   135,
    "mayda pak":     135,
    "mayda":         135,
    "small packet":  135,
    "small parcel":  135,
    "small":         135,
    "kichik":        135,
    "кичик":         135,
    "маленький":     135,
    "mini":          135,
    "мп":            135,
    "mp":            135,
    "m paket":       135,
    "kichik paket":  135,
    "little":        135,
    "tiny":          135,
}


def get_service_name(service_id: int) -> str:
    """ID bo'yicha xizmat nomini qaytaradi."""
    for s in SERVICES:
        if s["id"] == service_id:
            return s["name"]
    return "Noma'lum"


def get_service_emoji(service_id: int) -> str:
    """ID bo'yicha emoji qaytaradi."""
    for s in SERVICES:
        if s["id"] == service_id:
            return s["emoji"]
    return "📦"


def match_service(text: str) -> int | None:
    from difflib import SequenceMatcher

    if not text:
        return None

    t = text.lower().strip()

    # 🔢 Raqamli kiritmalarni aniqlash
    if t in ("1", "2", "3", "4"):
        # Xizmatlar tartibi: Bir qadam, Posilka, Xat, Mayda paket
        numeric_map = {
            "1": 209,   # Bir qadam
            "2": 136,   # Posilka
            "3": 211,   # Xat
            "4": 212,   # Mayda paket
        }
        return numeric_map.get(t)

    # 1. Exact
    if t in SERVICE_SYNONYMS:
        return SERVICE_SYNONYMS[t]

    # 2. Partial
    for syn in sorted(SERVICE_SYNONYMS, key=len, reverse=True):
        if len(syn) >= 3 and syn in t:
            return SERVICE_SYNONYMS[syn]

    # 3. Fuzzy
    best_score = 0.0
    best_id = None
    words = t.split()
    for word in words:
        if len(word) < 3:
            continue
        for syn, sid in SERVICE_SYNONYMS.items():
            if abs(len(word) - len(syn)) > 4:
                continue
            score = SequenceMatcher(None, word, syn).ratio()
            if score > best_score and score >= 0.82:
                best_score = score
                best_id = sid

    return best_id