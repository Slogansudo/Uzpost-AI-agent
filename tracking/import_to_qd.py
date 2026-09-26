import os
import json
import uuid
import asyncio
import argparse
import re
import aiohttp
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel

load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION     = "tracking"
EMBED_DIM      = 1024
BATCH_SIZE     = 20

# E5 modeli — normalize_embeddings=True MAJBURIY
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


# ─── MATN TOZALASH ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """URL, ortiqcha bo'sh joylarni olib tashlaydi."""
    if not text:
        return ""
    # URL ni olib tashlaymiz
    text = re.sub(r'https?://\S+', '', text)
    # "Batafsil:", "Learn more:", "Details:" kabi oxirgi so'zlarni olib tashlaymiz
    text = re.sub(r'(Batafsil|Learn more|More info|Details|Подробнее)[:\.]?\s*$', '', text, flags=re.IGNORECASE)
    # Ortiqcha bo'sh joylar
    text = ' '.join(text.split())
    return text.strip()


def build_passage_text(item: dict) -> str:
    """
    E5 modeli uchun to'g'ri "passage: ..." formati.

    Faqat EN ishlatamiz — eng aniq natija shu yo'l bilan.
    Savol + javob birlashtiriladi → kontekstli matn.
    """
    q_en = clean_text(item.get("question_en", ""))
    a_en = clean_text(item.get("answer_en",   ""))

    # Kategoriyani ham qo'shamiz — disambiguation uchun
    category = item.get("category", "").replace("_", " ")

    parts = []
    if category:
        parts.append(f"Category: {category}.")
    if q_en:
        parts.append(f"Question: {q_en}")
    if a_en:
        parts.append(f"Answer: {a_en}")

    # E5 uchun "passage: " prefiksi MAJBURIY
    return "passage: " + " ".join(parts)


# ─── EMBEDDING ────────────────────────────────────────────────────────────────

def get_embedding_sync(text: str) -> list[float]:
    result = model.encode(
        [text],
        batch_size=1,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return result["dense_vecs"][0].tolist()


# ─── QDRANT ───────────────────────────────────────────────────────────────────

def get_headers():
    h = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        h["api-key"] = QDRANT_API_KEY
    return h


async def delete_collection(session: aiohttp.ClientSession):
    async with session.delete(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers=get_headers(),
    ) as resp:
        if resp.status in (200, 404):
            print(f"[Qdrant] '{COLLECTION}' o'chirildi.")


async def ensure_collection(session: aiohttp.ClientSession):
    # Mavjudligini tekshirish
    async with session.get(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers=get_headers(),
    ) as resp:
        if resp.status == 200:
            data = await resp.json()
            existing_dim = data["result"]["config"]["params"]["vectors"]["size"]
            if existing_dim != EMBED_DIM:
                print(f"  ⚠ Dimension mismatch: bazada {existing_dim}, kerak {EMBED_DIM}")
                print(f"  → Collection o'chiriladi va qayta yaratiladi...")
                await delete_collection(session)
            else:
                print(f"[Qdrant] '{COLLECTION}' mavjud (dim={existing_dim}). ✓")
                return

    # Yaratish
    print(f"[Qdrant] '{COLLECTION}' yaratilmoqda (dim={EMBED_DIM})...")
    async with session.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers=get_headers(),
        json={
            "vectors": {
                "size": EMBED_DIM,
                "distance": "Cosine",
            },
            "optimizers_config": {
                "indexing_threshold": 0  # darhol index qilish
            }
        },
    ) as resp:
        if resp.status not in (200, 201):
            text = await resp.text()
            raise RuntimeError(f"Collection yaratishda xato: {text}")
    print(f"[Qdrant] '{COLLECTION}' yaratildi. ✓")


async def upsert_batch(session: aiohttp.ClientSession, points: list) -> bool:
    async with session.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        headers=get_headers(),
        json={"points": points},
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        if resp.status not in (200, 201):
            print(f"  [Upsert xato] {await resp.text()[:200]}")
            return False
        return True


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def main(json_file: str, reset: bool):
    print(f"\n{'='*60}")
    print(f"  UzPost Qdrant Import — TO'G'RILANGAN VERSIYA")
    print(f"  Model  : intfloat/multilingual-e5-large")
    print(f"  Fayl   : {json_file}")
    print(f"  Prefix : 'passage: ...' (import) | 'query: ...' (search)")
    print(f"{'='*60}\n")

    with open(json_file, encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"[Load] {len(dataset)} ta element yuklandi.")

    # Duplicate ID tekshirish
    ids = [item.get("id") for item in dataset]
    if len(ids) != len(set(ids)):
        dupe_count = len(ids) - len(set(ids))
        print(f"  ⚠ {dupe_count} ta duplicate ID — UUID ishlatiladi.")
        use_uuid = True
    else:
        use_uuid = False

    async with aiohttp.ClientSession() as session:
        if reset:
            await delete_collection(session)
        await ensure_collection(session)

        batch     = []
        ok_count  = 0
        err_count = 0

        print(f"\n[Import] Embedding va yuklash boshlanyapdi...\n")

        for i, item in enumerate(dataset, 1):
            # Passage matni tuzamiz
            passage = build_passage_text(item)

            # Embedding
            try:
                vector = get_embedding_sync(passage)
            except Exception as e:
                print(f"  ✗ [{i}] Embed xato: {e}")
                err_count += 1
                continue

            # Unique point ID
            if use_uuid:
                point_id = str(uuid.uuid4())
            else:
                point_id = int(item["id"])

            point = {
                "id"     : point_id,
                "vector" : vector,
                "payload": {
                    "source_id"   : item.get("id"),
                    "category"    : item.get("category", ""),
                    "question_uz" : item.get("question_uz", ""),
                    "question_ru" : item.get("question_ru", ""),
                    "question_en" : item.get("question_en", ""),
                    "answer_uz"   : item.get("answer_uz",   ""),
                    "answer_ru"   : item.get("answer_ru",   ""),
                    "answer_en"   : item.get("answer_en",   ""),
                    "passage"     : passage,  # debug uchun
                },
            }
            batch.append(point)

            if len(batch) >= BATCH_SIZE or i == len(dataset):
                print(f"  [{i}/{len(dataset)}] Upsert {len(batch)} ta...", end=" ")
                ok = await upsert_batch(session, batch)
                if ok:
                    ok_count += len(batch)
                    print(f"✓  (jami: {ok_count})")
                else:
                    err_count += len(batch)
                    print("✗")
                batch = []

            await asyncio.sleep(0.02)

        print(f"\n{'='*60}")
        print(f"  ✓ Muvaffaqiyatli: {ok_count}")
        print(f"  ✗ Xato          : {err_count}")
        print(f"  Collection      : {QDRANT_URL}/collections/{COLLECTION}")
        print(f"{'='*60}")

        # Avtomatik test
        print("\n[Test] Import tekshirilmoqda...\n")
        await _test_search(session)


async def _test_search(session: aiohttp.ClientSession):
    """Import to'g'ri bo'lganini tekshirish uchun 3 ta test."""
    test_queries = [
        ("What is courier service?",          "courier"),
        ("Bir Qadam express postal service",   "bir_qadam_service"),
        ("letter maximum weight grams",         "letters"),
    ]

    for query_en, expected_cat in test_queries:
        # TO'G'RI: "query: " prefiksi bilan encode
        res = model.encode(["query: " + query_en], return_dense=True)
        vec = res["dense_vecs"][0].tolist()

        async with session.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
            headers=get_headers(),
            json={"vector": vec, "limit": 1, "with_payload": True},
        ) as resp:
            data = await resp.json()
            results = data.get("result", [])

        if results:
            r   = results[0]
            cat = r["payload"].get("category", "")
            sc  = r["score"]
            ok  = "✅" if cat == expected_cat else "⚠"
            print(f"  {ok} Query: '{query_en}'")
            print(f"     Score : {sc:.4f}")
            print(f"     Cat   : {cat} (kutilgan: {expected_cat})")
            print(f"     Q_en  : {r['payload'].get('question_en', '')[:70]}")
        else:
            print(f"  ✗ '{query_en}' → natija yo'q")
        print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  default="tracking/tracking.json")
    parser.add_argument("--reset", action="store_true",
                        help="Collection ni o'chirib qayta yaratish")
    args = parser.parse_args()
    asyncio.run(main(args.file, args.reset))