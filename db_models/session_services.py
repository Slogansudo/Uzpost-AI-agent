
import os
import asyncio
from datetime import datetime, timezone
from typing import Optional
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
# Misol: postgresql://user:password@localhost:5432/uzpost_db

_pool: Optional[asyncpg.Pool] = None


# ============================================================
# POOL — startup da bir marta yaratiladi
# ============================================================
async def init_db_pool() -> None:
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=2,
        max_size=10,
    )
    # Jadval yo'q bo'lsa yaratish
    async with _pool.acquire() as conn:
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id         SERIAL PRIMARY KEY,
                    user_key   VARCHAR(64)  NOT NULL,
                    role       VARCHAR(16)  NOT NULL,
                    content    TEXT         NOT NULL,
                    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_chat_history_user_key
                    ON chat_history (user_key, created_at DESC);

                CREATE TABLE IF NOT EXISTS global_qa_cache (
                    id         SERIAL PRIMARY KEY,
                    query      TEXT         NOT NULL UNIQUE,
                    answer     TEXT         NOT NULL,
                    intent     VARCHAR(20)  NOT NULL,
                    created_at TIMESTAMPTZ  NOT NULL DEFAULT NOW()
                );
            """)
    print("[db] chat_history jadvali tayyor")


# ============================================================
# YOZISH
# ============================================================
async def add_message(user_key: str, role: str, content: str) -> None:
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO chat_history (user_key, role, content) VALUES ($1, $2, $3)",
            user_key, role, content,
        )


# ============================================================
# O'QISH
# ============================================================
from datetime import datetime, date, time

async def get_history(user_key: str, limit: int = 20) -> list[dict]:
    """
    Bugungi kun bo‘yicha oxirgi `limit` ta xabar (eski → yangi)
    """
    if not _pool:
        return []

    start_of_day = datetime.combine(date.today(), time.min)
    end_of_day = datetime.combine(date.today(), time.max)

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content, created_at
            FROM (
                SELECT role, content, created_at
                FROM chat_history
                WHERE user_key = $1
                  AND created_at BETWEEN $2 AND $3
                ORDER BY created_at DESC
                LIMIT $4
            ) sub
            ORDER BY created_at ASC
            """,
            user_key,
            start_of_day,
            end_of_day,
            limit,
        )

    return [
        {
            "role": r["role"],
            "content": r["content"],
            "ts": r["created_at"].strftime("%Y-%m-%d %H:%M"),
        }
        for r in rows
    ]


async def get_context_messages(user_key: str, limit: int = 10) -> list[dict]:
    """
    Groq ga yuborish uchun oxirgi `limit` ta xabar.
    Faqat {role, content} qaytaradi.
    """
    history = await get_history(user_key, limit=limit)
    return [{"role": h["role"], "content": h["content"]} for h in history]


async def clear_history(user_key: str) -> None:
    """Foydalanuvchi tarixini tozalaydi."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM chat_history WHERE user_key = $1",
            user_key,
        )



async def get_global_cache(query: str) -> Optional[str]:
    """Global cache dan javob olish."""
    if not _pool:
        return None
    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT answer FROM global_qa_cache WHERE query = $1",
            query.strip(),
        )
    return row["answer"] if row else None


async def set_global_cache(query: str, answer: str, intent: str) -> None:
    """Global cache ga saqlash — conflict bo'lsa o'tkazib yuborish."""
    if not _pool:
        return
    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO global_qa_cache (query, answer, intent)
            VALUES ($1, $2, $3)
            ON CONFLICT (query) DO NOTHING
            """,
            query.strip(), answer, intent,
        )