"""
tracking/api.py — Faqat kerakli maydonlar
==========================================
Qaytaradi:
  found, barcode, from_addr, to_addr, events[{date, status_uz, status_ru, status_en, place}]
"""

import re
import asyncio
import requests
from datetime import datetime, timedelta

_TRACKING_URL    = "https://tracking.pochta.uz/api/v1/public/test_ai/aibot/{barcode}/"
_TIMEZONE_OFFSET = timedelta(hours=5)

_BARCODE_RE = re.compile(r"\b([A-Z]{2,4}\d{7,14}[A-Z]{0,2})\b", re.IGNORECASE)
_MIN_LEN, _MAX_LEN = 10, 22


def extract_barcode(text: str) -> str | None:
    if not text:
        return None
    for c in _BARCODE_RE.findall(text.upper()):
        if _MIN_LEN <= len(c) <= _MAX_LEN:
            letters = sum(ch.isalpha() for ch in c)
            digits  = sum(ch.isdigit() for ch in c)
            if letters >= 2 and digits >= 7:
                return c
    return None


def _fmt_date(raw: str) -> str:
    try:
        if raw.endswith("Z"):
            raw = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw) + _TIMEZONE_OFFSET
        return dt.strftime("%d.%m.%Y %H:%M")
    except Exception:
        return raw


def _loc_str(loc: dict) -> str:
    addr    = (loc.get("address") or "").strip()[:60]
    country = ((loc.get("country") or {}).get("name") or "").strip()
    return ", ".join(filter(None, [addr, country]))


async def fetch_tracking_data(barcode: str) -> dict:
    """
    Qaytaradi:
    {
      barcode, found,
      from_addr,   # "1 Mustaqillik ko'chasi, Toshkent, Uzbekistan"
      to_addr,     # "Do'stlik ul. Gurlan 48, Uzbekistan"
      events: [
        { date, status_uz, status_ru, status_en, place }
        ...  # yangi → eski tartibda
      ]
    }
    """
    try:
        url  = _TRACKING_URL.format(barcode=barcode)
        resp = await asyncio.to_thread(requests.get, url, timeout=6)
        if resp.status_code != 200:
            return {"barcode": barcode, "found": False}

        raw = resp.json()

        # ── Manzillar ──
        locations = ((raw.get("header") or {}).get("data") or {}).get("locations") or []
        from_addr = to_addr = ""
        for loc in locations:
            s = _loc_str(loc)
            if loc.get("pickup"):
                from_addr = s
            else:
                to_addr = s

        # ── Voqealar ──
        items = ((raw.get("shipox") or {}).get("data") or {}).get("list") or []
        if not items:
            return {"barcode": barcode, "found": False}

        events = []
        for item in items:
            wh    = item.get("warehouse") or {}
            place = wh.get("public_name") or wh.get("name") or ""
            events.append({
                "date":       _fmt_date(item.get("date", "")),
                "status_uz":  item.get("status_uz", ""),
                "status_ru":  item.get("status_ru", ""),
                "status_en":  item.get("status_eng", ""),
                "place":      place,
            })

        return {
            "barcode":   barcode,
            "found":     True,
            "from_addr": from_addr,
            "to_addr":   to_addr,
            "events":    events,   # [0] = eng so'nggi
        }

    except Exception as e:
        print(f"[Tracking API] barcode={barcode} error={e}")
        return {"barcode": barcode, "found": False}

