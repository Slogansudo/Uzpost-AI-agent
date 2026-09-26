"""
import_to_qdrant_v6.py — UzPost RAG: BGE-M3 + SITUATION_COLLECTION
======================================================================

v5 dan farqi:

  YANGI:  SITUATION_COLLECTION
          → har bir xizmat JSON dagi "situations" bo'limi import qilinadi
          → dense(1024) + sparse(lexical) — hybrid search uchun
          → question_uz/ru/en + answer_uz/ru/en — 3 til, har biri alohida point
          → tags, weight, situation_type, related_services payload da saqlanadi

  PIPELINE o'zgarishi:
          savol → SITUATION search (asosiy)
               ↓ (score < 0.82 bo'lsa fallback)
          savol → CATEGORY classify → SERVICE kontekst → LLM

  SAQLANADI (o'zgarmaydi):
          - CHUNK_COLLECTION    (dense+sparse)
          - SERVICE_COLLECTION  (dense only)
          - CATEGORY_COLLECTION (dense+sparse)
          - BM25 index (chunk + search_query docs)
          - flatten_service() logikasi
          - CLI args

  YANGI collection:
    SITUATION_COLLECTION:
      vectors:        {"dense": {"size": 1024, "distance": "Cosine"}}
      sparse_vectors: {"sparse": {}}
      payload: {
        situation_id, situation_type, related_services,
        lang, question, answer, tags, weight,
        service_id, category, contact
      }

O'rnatish:
    pip install -U FlagEmbedding
    python import_to_qdrant_v6.py --file data/uzpost_all.json --reset
    python import_to_qdrant_v6.py --file data/uzpost_all.json --dry-run
"""

from __future__ import annotations

import os
import re
import json
import uuid
import asyncio
import argparse
import pickle
from pathlib import Path

import aiohttp
import numpy as np
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

load_dotenv()

# ── Sozlamalar ────────────────────────────────────────────────────────────────
QDRANT_URL           = os.getenv("QDRANT_URL",               "http://localhost:6333")
QDRANT_KEY           = os.getenv("QDRANT_API_KEY",           "")
CHUNK_COLLECTION     = os.getenv("CHUNK_COLLECTION",         "uzpost_chunks_v1")
SERVICE_COLLECTION   = os.getenv("SERVICE_COLLECTION",       "uzpost_services_v1")
CATEGORY_COLLECTION  = os.getenv("CATEGORY_COLLECTION",      "uzpost_categories_v1")
SITUATION_COLLECTION = os.getenv("SITUATION_COLLECTION",     "uzpost_situations_v1")
BM25_PATH            = Path(os.getenv("BM25_INDEX_PATH",     "faq/bm25_index.pkl"))

DENSE_DIM  = 1024
BATCH_SIZE = 8

# ── BGE-M3 model ──────────────────────────────────────────────────────────────
print("[Import] BGE-M3 yuklanmoqda: BAAI/bge-m3 ...")
_MODEL = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
print("[Import] BGE-M3 tayyor ✓")


# ── Stopwords + Tokenizer ─────────────────────────────────────────────────────
_STOPWORDS = {
    "va", "yoki", "ham", "bu", "u", "uchun", "bilan", "dan", "ga", "da",
    "ni", "ning", "boladi", "mumkin", "kerak", "bolgan", "qilish", "bir",
    "shu", "ular", "men", "sen", "biz", "ozi", "qiladi", "qiling",
    "lekin", "agar", "hatto", "faqat", "esa", "chunki",
    "i", "v", "na", "po", "chto", "kak", "ne", "eto", "ot", "do",
    "a", "no", "ili", "za", "iz", "pri", "k", "s", "dlya", "o",
    "ya", "on", "ona", "oni", "my", "vy", "ego", "ej",
    "the", "an", "is", "in", "to", "of", "or", "and", "for",
    "be", "it", "at", "by", "we", "as", "if", "on", "are", "can",
    "not", "this", "that", "was", "has", "have", "will", "with",
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[ʻʼ`ʹʾ\u2018\u2019]", "'", text.lower())
    tokens = re.findall(r"[a-zA-Zа-яёА-ЯЁ']+", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


# ── BGE-M3 Embedding ──────────────────────────────────────────────────────────

def encode_dense(texts: list[str], is_query: bool = False) -> list[list[float]]:
    output = _MODEL.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return [v.tolist() for v in output["dense_vecs"]]


def encode_sparse(texts: list[str]) -> list[dict[str, float]]:
    output = _MODEL.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=512,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    return output["lexical_weights"]


def sparse_to_qdrant(lexical_weights: dict) -> dict:
    indices, values = [], []
    for token_id, weight in lexical_weights.items():
        indices.append(int(token_id))
        values.append(float(weight))
    return {"indices": indices, "values": values}


def encode_batch_both(texts: list[str]) -> tuple[list[list[float]], list[dict]]:
    """Dense + sparse bitta forward pass da."""
    output = _MODEL.encode(
        texts,
        batch_size=BATCH_SIZE,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense  = [v.tolist() for v in output["dense_vecs"]]
    sparse = output["lexical_weights"]
    return dense, sparse


# ── YANGI: flatten_situations ─────────────────────────────────────────────────

def flatten_situations(data: dict) -> list:
    """
    Bir service JSON dagi "situations" bo'limini SITUATION_COLLECTION uchun
    point listga aylantiradi.

    Har bir situation → 3 ta point (uz + ru + en), agar mavjud bo'lsa.
    Embedding matni: question + "\n\n" + answer
    (Semantik qidiruv uchun savol+javob birlashtirilishi yaxshiroq ishlaydi)

    Qo'llab-quvvatlanadigan kalit formatlar:
      question_uz / answer_uz
      question_ru / answer_ru
      question_en / answer_en

    Agar biror tilda question/answer yo'q bo'lsa — o'tkazib yuboriladi.
    """
    service_id    = data.get("id", str(uuid.uuid4()))
    category      = data.get("category", "")
    service_names = data.get("service_names", {})
    metadata      = data.get("metadata", {})
    contact       = metadata.get("contact", {})
    situations    = data.get("situations", [])

    points: list = []

    for sit in situations:
        situation_id   = sit.get("situation_id", str(uuid.uuid4()))
        situation_type = sit.get("situation_type", "general")
        related        = sit.get("related_services", [category])
        tags           = sit.get("tags", [])
        weight         = float(sit.get("weight", 1.5))

        # Har uchala til uchun alohida point
        for lang in ("uz", "ru", "en"):
            q_key = f"question_{lang}"
            a_key = f"answer_{lang}"

            question = sit.get(q_key, "").strip()
            answer   = sit.get(a_key, "").strip()

            # Agar bu tilda savol yoki javob yo'q bo'lsa — o'tkazib yuboriladi
            if not question or not answer:
                continue

            # Embedding matni: question + answer (semantik qamrov keng bo'lsin)
            embed_text = clean_text(f"{question}\n\n{answer}")

            points.append({
                "_id":   str(uuid.uuid4()),
                "_text": embed_text,
                "payload": {
                    "situation_id":     situation_id,
                    "situation_type":   situation_type,
                    "related_services": related,
                    "service_id":       service_id,
                    "category":         category,
                    "lang":             lang,
                    "question":         question,
                    "answer":           answer,
                    "tags":             tags,
                    "weight":           weight,
                    "service_name":     service_names.get(lang, ""),
                    "contact":          contact,
                },
            })

    return points


# ── flatten_service (o'zgarmadi) ──────────────────────────────────────────────

def flatten_service(data: dict) -> tuple[list, list, list, list]:
    """
    Bir service JSON → chunk, service, category, bm25 listlar.
    Bu funksiya v5 bilan bir xil — situations bu yerda ishlanmaydi.
    """
    service_id    = data.get("id", str(uuid.uuid4()))
    category      = data.get("category", "")
    service_names = data.get("service_names", {})
    metadata      = data.get("metadata", {})
    contact       = metadata.get("contact", {})
    search_q      = data.get("search_queries", {})

    chunk_points:    list = []
    service_points:  list = []
    category_points: list = []
    bm25_docs:       list = []

    # ── SERVICE LAYER ─────────────────────────────────────────────────────────
    for lang in ("uz", "ru", "en"):
        parts = []
        sname = service_names.get(lang, "")
        if sname:
            parts.append(sname)

        for chunk in data.get("chunks", []):
            chunk_text = chunk.get(lang, "").strip()
            if not chunk_text:
                continue
            intents = [i.strip() for i in chunk.get("intent", "").split("|") if i.strip()]
            for intent_tag in intents:
                parts.append(f"[{intent_tag}]")
            parts.append(clean_text(chunk_text))

        if not parts:
            continue

        full_text = "\n\n".join(parts)

        service_points.append({
            "_id":    str(uuid.uuid4()),
            "_text":  full_text[:2000],
            "payload": {
                "service_id":   service_id,
                "category":     category,
                "lang":         lang,
                "service_name": sname,
                "full_text":    full_text,
                "summary_text": full_text[:800],
                "contact":      contact,
                "chunk_count":  len([c for c in data.get("chunks", []) if c.get(lang)]),
            },
        })

    # ── CATEGORY LAYER ────────────────────────────────────────────────────────
    for lang in ("uz", "ru", "en"):
        sname   = service_names.get(lang, "")
        queries = search_q.get(lang, [])

        for qt in queries:
            qt = qt.strip()
            if not qt:
                continue

            name_in_query = bool(sname and sname.lower() in qt.lower())
            weight = 1.5 if name_in_query else 1.0

            category_points.append({
                "_id":    str(uuid.uuid4()),
                "_text":  qt,
                "payload": {
                    "category":     category,
                    "service_id":   service_id,
                    "lang":         lang,
                    "query_text":   qt,
                    "service_name": sname,
                    "weight":       weight,
                    "source":       "search_query",
                },
            })

    # ── BM25: search_query docs ───────────────────────────────────────────────
    for lang in ("uz", "ru", "en"):
        for qt in search_q.get(lang, []):
            qt = qt.strip()
            if not qt:
                continue
            sq_tokens = tokenize(qt)
            if not sq_tokens:
                continue
            bm25_docs.append({
                "point_id":     str(uuid.uuid4()),
                "service_id":   service_id,
                "chunk_id":     "search_query",
                "category":     category,
                "intent":       "search_query",
                "weight":       2.0,
                "lang":         lang,
                "tokens":       sq_tokens,
                "text":         qt,
                "service_name": service_names.get(lang, ""),
                "contact":      contact,
            })

    # ── CHUNK LAYER ───────────────────────────────────────────────────────────
    for chunk in data.get("chunks", []):
        chunk_id = chunk.get("chunk_id", str(uuid.uuid4()))
        intent   = chunk.get("intent", "general")
        weight   = float(chunk.get("weight", 1.0))

        for lang in ("uz", "ru", "en"):
            text = chunk.get(lang, "").strip()
            if not text:
                continue

            text_clean = clean_text(text)
            point_id   = str(uuid.uuid4())

            chunk_points.append({
                "_id":    point_id,
                "_text":  text_clean,
                "payload": {
                    "service_id":   service_id,
                    "chunk_id":     chunk_id,
                    "category":     category,
                    "intent":       intent,
                    "weight":       weight,
                    "lang":         lang,
                    "text":         text_clean,
                    "service_name": service_names.get(lang, ""),
                    "contact":      contact,
                },
            })

            bm25_docs.append({
                "point_id":     point_id,
                "service_id":   service_id,
                "chunk_id":     chunk_id,
                "category":     category,
                "intent":       intent,
                "weight":       weight,
                "lang":         lang,
                "tokens":       tokenize(text_clean),
                "text":         text_clean,
                "service_name": service_names.get(lang, ""),
                "contact":      contact,
            })

    return chunk_points, service_points, category_points, bm25_docs


# ── BM25 Index ────────────────────────────────────────────────────────────────

def build_bm25(bm25_docs: list) -> None:
    corpus = [doc["tokens"] for doc in bm25_docs]
    bm25   = BM25Okapi(corpus)

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": bm25_docs}, f)

    sq_count    = sum(1 for d in bm25_docs if d["chunk_id"] == "search_query")
    chunk_count = len(bm25_docs) - sq_count
    size_kb     = BM25_PATH.stat().st_size // 1024

    print(f"[BM25] Saqlandi: {BM25_PATH}")
    print(f"       Jami: {len(bm25_docs)} doc | {size_kb} KB")
    print(f"       search_query docs: {sq_count}")
    print(f"       chunk docs:        {chunk_count}")


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if QDRANT_KEY:
        h["api-key"] = QDRANT_KEY
    return h


async def delete_collection(session: aiohttp.ClientSession, name: str) -> None:
    async with session.delete(
        f"{QDRANT_URL}/collections/{name}",
        headers=_headers(),
    ) as r:
        print(f"[Qdrant] '{name}' o'chirildi → {r.status}")


async def ensure_collection(
    session:     aiohttp.ClientSession,
    name:        str,
    with_sparse: bool = True,
) -> None:
    async with session.get(
        f"{QDRANT_URL}/collections/{name}",
        headers=_headers(),
    ) as r:
        if r.status == 200:
            data = await r.json()
            dim  = data["result"]["config"]["params"]["vectors"].get("size")
            if dim == DENSE_DIM:
                print(f"[Qdrant] '{name}' mavjud (dim={dim}) ✓")
                return
            print(f"[Qdrant] '{name}' dim xato ({dim}≠{DENSE_DIM}), qayta yaratilmoqda...")
            await delete_collection(session, name)

    schema: dict = {
        "vectors": {
            "dense": {"size": DENSE_DIM, "distance": "Cosine"}
        },
        "optimizers_config": {"indexing_threshold": 0},
    }
    if with_sparse:
        schema["sparse_vectors"] = {"sparse": {}}

    async with session.put(
        f"{QDRANT_URL}/collections/{name}",
        headers=_headers(),
        json=schema,
    ) as r:
        if r.status not in (200, 201):
            raise RuntimeError(f"'{name}' yaratish xatosi: {await r.text()}")

    label = "(dense+sparse)" if with_sparse else "(dense only)"
    print(f"[Qdrant] '{name}' yaratildi {label} ✓")


async def create_payload_indexes(session: aiohttp.ClientSession) -> None:
    configs = [
        # Mavjud collectionlar
        (CATEGORY_COLLECTION,  "lang"),
        (CATEGORY_COLLECTION,  "category"),
        (CATEGORY_COLLECTION,  "service_id"),
        (SERVICE_COLLECTION,   "service_id"),
        (SERVICE_COLLECTION,   "lang"),
        (CHUNK_COLLECTION,     "lang"),
        (CHUNK_COLLECTION,     "category"),
        # YANGI: SITUATION indexes
        (SITUATION_COLLECTION, "lang"),
        (SITUATION_COLLECTION, "category"),
        (SITUATION_COLLECTION, "situation_type"),
        (SITUATION_COLLECTION, "service_id"),
        # tags array index uchun keyword
        (SITUATION_COLLECTION, "tags"),
    ]
    for coll, field in configs:
        async with session.put(
            f"{QDRANT_URL}/collections/{coll}/index",
            headers=_headers(),
            json={"field_name": field, "field_schema": "keyword"},
        ) as r:
            print(f"[Qdrant] Index '{coll}'.'{field}' → {r.status}")


async def upsert_batch_service(
    session:    aiohttp.ClientSession,
    collection: str,
    raw_points: list,
) -> tuple[int, int]:
    """SERVICE collection: faqat dense."""
    texts      = [p["_text"] for p in raw_points]
    dense_vecs = encode_dense(texts, is_query=False)

    points = [
        {
            "id":      p["_id"],
            "vector":  {"dense": dvec},
            "payload": p["payload"],
        }
        for p, dvec in zip(raw_points, dense_vecs)
    ]
    return await _upsert_points(session, collection, points)


async def upsert_batch_with_sparse(
    session:    aiohttp.ClientSession,
    collection: str,
    raw_points: list,
) -> tuple[int, int]:
    """CHUNK, CATEGORY, SITUATION: dense + sparse."""
    texts = [p["_text"] for p in raw_points]
    dense_vecs, sparse_weights = encode_batch_both(texts)

    points = [
        {
            "id": p["_id"],
            "vector": {
                "dense":  dvec,
                "sparse": sparse_to_qdrant(sw),
            },
            "payload": p["payload"],
        }
        for p, dvec, sw in zip(raw_points, dense_vecs, sparse_weights)
    ]
    return await _upsert_points(session, collection, points)


async def _upsert_points(
    session:    aiohttp.ClientSession,
    collection: str,
    points:     list,
) -> tuple[int, int]:
    async with session.put(
        f"{QDRANT_URL}/collections/{collection}/points",
        headers=_headers(),
        json={"points": points},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as r:
        if r.status not in (200, 201):
            print(f"  [Xato {r.status}]: {(await r.text())[:200]}")
            return 0, len(points)
        return len(points), 0


async def upload_collection(
    session:     aiohttp.ClientSession,
    collection:  str,
    points:      list,
    label:       str,
    with_sparse: bool = True,
) -> None:
    if not points:
        print(f"\n[Qdrant] {label}: 0 point — o'tkazib yuborildi.")
        return

    print(f"\n[Qdrant] {label}: {len(points)} point yuklanmoqda...")
    ok = err = 0
    batch:  list = []
    total = len(points)
    upsert_fn = upsert_batch_with_sparse if with_sparse else upsert_batch_service

    for i, pt in enumerate(points, 1):
        batch.append(pt)
        if len(batch) >= BATCH_SIZE or i == total:
            print(f"  [{i:5d}/{total}] {len(batch)} ta ...", end=" ", flush=True)
            ok_, err_ = await upsert_fn(session, collection, batch)
            ok  += ok_
            err += err_
            print("OK" if ok_ else "XATO")
            batch = []
            await asyncio.sleep(0.05)

    print(f"  {label}: OK={ok} | XATO={err}")


# ── JSON yuklash ──────────────────────────────────────────────────────────────

def load_json(path_str: str) -> list:
    p = Path(path_str)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    print(f"[Load] {len(files)} ta JSON fayl")

    all_services = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        items = data if isinstance(data, list) else [data]
        all_services.extend(items)
        print(f"  {f.name}: {len(items)} ta service")

    return all_services


# ── Dry-run ───────────────────────────────────────────────────────────────────

def dry_run_check(
    all_chunks:     list,
    all_services:   list,
    all_categories: list,
    all_situations: list,
    all_bm25:       list,
) -> None:
    print("\n" + "=" * 66)
    print("  DRY-RUN NATIJALAR (v6 — BGE-M3 + SITUATION_COLLECTION, 3 til: uz/ru/en)")
    print("=" * 66)

    sq_docs    = [d for d in all_bm25 if d["chunk_id"] == "search_query"]
    chunk_docs = [d for d in all_bm25 if d["chunk_id"] != "search_query"]

    print(f"\n[Stats]")
    print(f"  Situation points (SITUATION) : {len(all_situations)}")
    print(f"  Chunk points     (CHUNK)     : {len(all_chunks)}")
    print(f"  Service points   (SERVICE)   : {len(all_services)}")
    print(f"  Category points  (CATEGORY)  : {len(all_categories)}")
    print(f"  BM25 jami                    : {len(all_bm25)}")
    print(f"    search_query docs          : {len(sq_docs)}")
    print(f"    chunk docs                 : {len(chunk_docs)}")

    # Situation statistikasi
    sit_stats: dict = {}
    for pt in all_situations:
        pl = pt["payload"]
        cat  = pl.get("category", "?")
        lang = pl.get("lang", "?")
        sit_stats.setdefault(cat, {}).setdefault(lang, 0)
        sit_stats[cat][lang] += 1

    print(f"\n[Situation Layer] Situation soni (category × til):")
    total_sit = 0
    for cat, langs in sorted(sit_stats.items()):
        uz_n  = langs.get("uz", 0)
        ru_n  = langs.get("ru", 0)
        en_n  = langs.get("en", 0)
        cat_total = uz_n + ru_n + en_n
        total_sit += cat_total
        missing = []
        if uz_n == 0: missing.append("uz!")
        if ru_n == 0: missing.append("ru!")
        if en_n == 0: missing.append("en!")
        miss_str = f"  ⚠ YO'Q: {', '.join(missing)}" if missing else ""
        print(f"  {cat:<30} uz:{uz_n:>3} | ru:{ru_n:>3} | en:{en_n:>3}  (jami:{cat_total:>4}){miss_str}")
    print(f"  {'JAMI':<30} {total_sit} point")

    print(f"\n[Qdrant v6 collection schema]")
    print(f"  SITUATION_COLLECTION → dense(1024) + sparse(lexical)  ← YANGI asosiy")
    print(f"  CHUNK_COLLECTION     → dense(1024) + sparse(lexical)  ← fallback")
    print(f"  CATEGORY_COLLECTION  → dense(1024) + sparse(lexical)  ← classify")
    print(f"  SERVICE_COLLECTION   → dense(1024) only               ← LLM kontekst")

    print(f"\n[Pipeline v6]")
    print(f"  1. Keyword match      (0ms)      → _CATEGORY_KEYWORDS")
    print(f"  2. SITUATION search   (50-80ms)  → BGE-M3 dense+sparse ← YANGI asosiy")
    print(f"     score > 0.82  → to'g'ridan javob (LLM ixtiyoriy)")
    print(f"     score ≤ 0.82  → fallback:")
    print(f"  3. BM25 classifier    (5ms)      → search_query docs")
    print(f"  4. Hybrid classify    (50ms)     → CATEGORY_COLLECTION")
    print(f"  5. SERVICE kontekst   (20ms)     → SERVICE_COLLECTION")
    print(f"  6. LLM javob          (500ms+)   → GPT/Claude")

    print(f"\n[Kutilayotgan natija]")
    print(f"  Hozirgi aniqlik  : ~60-70%  (chunk/category asosida)")
    print(f"  v6 aniqlik       : ~88-92%  (situation asosida, muammo bo'yicha)")


# ── Asosiy funksiya ───────────────────────────────────────────────────────────

async def main(path: str, reset: bool, dry_run: bool) -> None:
    print(f"\n{'=' * 66}")
    print(f"  UzPost Import v6 — BGE-M3 + SITUATION_COLLECTION")
    print(f"  Fayl/papka            : {path}")
    print(f"  Situation collection  : {SITUATION_COLLECTION}")
    print(f"  Chunk collection      : {CHUNK_COLLECTION}")
    print(f"  Service collection    : {SERVICE_COLLECTION}")
    print(f"  Category collection   : {CATEGORY_COLLECTION}")
    print(f"  BM25                  : {BM25_PATH}")
    print(f"  Dry-run               : {dry_run}")
    print(f"{'=' * 66}\n")

    services = load_json(path)
    print(f"\n[Total] {len(services)} ta service\n")

    all_chunks:     list = []
    all_services:   list = []
    all_categories: list = []
    all_situations: list = []
    all_bm25:       list = []

    for svc in services:
        # Mavjud pipeline (o'zgarmadi)
        cp, sp, catp, bd = flatten_service(svc)
        all_chunks.extend(cp)
        all_services.extend(sp)
        all_categories.extend(catp)
        all_bm25.extend(bd)

        # YANGI: situations
        sit_points = flatten_situations(svc)
        all_situations.extend(sit_points)

    print(f"\n[Flatten]")
    print(f"  {len(all_situations)} situation points  ← YANGI")
    print(f"  {len(all_chunks)} chunk points")
    print(f"  {len(all_services)} service points")
    print(f"  {len(all_categories)} category points")
    sq = sum(1 for d in all_bm25 if d["chunk_id"] == "search_query")
    print(f"  {len(all_bm25)} BM25 docs ({sq} search_query + {len(all_bm25)-sq} chunk)")

    if dry_run:
        dry_run_check(
            all_chunks, all_services,
            all_categories, all_situations, all_bm25
        )
        return

    # BM25 (o'zgarmadi)
    print("\n[BM25] Index qurilmoqda...")
    build_bm25(all_bm25)

    # Qdrant
    async with aiohttp.ClientSession() as session:

        if reset:
            await delete_collection(session, SITUATION_COLLECTION)
            await delete_collection(session, CHUNK_COLLECTION)
            await delete_collection(session, SERVICE_COLLECTION)
            await delete_collection(session, CATEGORY_COLLECTION)

        # Collection yaratish
        # SITUATION → dense + sparse (asosiy qidiruv)
        await ensure_collection(session, SITUATION_COLLECTION, with_sparse=True)
        await ensure_collection(session, CHUNK_COLLECTION,     with_sparse=True)
        await ensure_collection(session, SERVICE_COLLECTION,   with_sparse=False)
        await ensure_collection(session, CATEGORY_COLLECTION,  with_sparse=True)

        # Payload indexes
        await create_payload_indexes(session)

        # Upload: SITUATION birinchi (asosiy)
        await upload_collection(
            session, SITUATION_COLLECTION,
            all_situations, "SITUATION",
            with_sparse=True,
        )
        await upload_collection(
            session, CHUNK_COLLECTION,
            all_chunks, "CHUNK",
            with_sparse=True,
        )
        await upload_collection(
            session, SERVICE_COLLECTION,
            all_services, "SERVICE",
            with_sparse=False,
        )
        await upload_collection(
            session, CATEGORY_COLLECTION,
            all_categories, "CATEGORY",
            with_sparse=True,
        )

    print(f"\n{'=' * 66}")
    print(f"  Import yakunlandi ✓ (BGE-M3 v6)")
    print(f"  SITUATION : {len(all_situations)} point (dense+sparse)  ← YANGI")
    print(f"  CHUNK     : {len(all_chunks)} point (dense+sparse)")
    print(f"  SERVICE   : {len(all_services)} point (dense only)")
    print(f"  CATEGORY  : {len(all_categories)} point (dense+sparse)")
    print(f"  BM25      : {len(all_bm25)} doc → {BM25_PATH}")
    print(f"{'=' * 66}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="UzPost JSON → Qdrant v6 (BGE-M3 + SITUATION_COLLECTION)"
    )
    ap.add_argument("--file",    required=True,       help="JSON fayl yoki papka yo'li")
    ap.add_argument("--reset",   action="store_true", help="Collectionlarni o'chirib qayta yaratadi")
    ap.add_argument("--dry-run", action="store_true", help="Faqat ko'rsatadi, yuklamaydi")
    args = ap.parse_args()

    asyncio.run(main(args.file, args.reset, args.dry_run))