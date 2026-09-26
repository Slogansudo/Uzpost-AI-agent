"""
admin.py — UzPost AI Admin Panel API
=====================================
Endpoints:
  POST /admin/login          — token olish
  GET  /admin/stats          — umumiy statistika
  GET  /admin/chats          — foydalanuvchilar ro'yxati (paginated)
  GET  /admin/chats/{user_key} — bitta foydalanuvchi tarixi (paginated)
  DELETE /admin/chats/{user_key} — bitta foydalanuvchi tarixini o'chirish
  DELETE /admin/chats          — barcha chat tarixini o'chirish (?confirm=true)
  GET  /admin/cache          — global_qa_cache ro'yxati (paginated)
  DELETE /admin/cache/{id}   — cache yozuvini o'chirish
"""

import os
import hashlib
import time
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncpg
from fastapi import APIRouter

load_dotenv()

# ─── CONFIG ────────────────────────────────────────────────────────────────────

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "uzpost2024")
DATABASE_URL   = os.getenv("DATABASE_URL")

# Oddiy in-memory token (production da JWT ishlatish tavsiya etiladi)
_VALID_TOKENS: dict[str, float] = {}   # token → expire_time
TOKEN_TTL = 60 * 60 * 8               # 8 soat

# ─── APP ───────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/admin", tags=["Admin"])


# ─── DB POOL ───────────────────────────────────────────────────────────────────

_pool: Optional[asyncpg.Pool] = None

def set_pool(pool: asyncpg.Pool) -> None:
    global _pool
    _pool = pool

async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Admin pool init qilinmagan")
    return _pool





# ─── AUTH ──────────────────────────────────────────────────────────────────────

security = HTTPBearer()


def _make_token(username: str) -> str:
    raw = f"{username}:{time.time()}:{ADMIN_PASSWORD}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _verify_token(token: str) -> bool:
    entry = _VALID_TOKENS.get(token)
    if not entry:
        return False
    if time.time() > entry:
        del _VALID_TOKENS[token]
        return False
    return True


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if not _verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Token noto'g'ri yoki muddati o'tgan")
    return credentials.credentials


# ─── SCHEMAS ───────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token:      str
    expires_in: int   # soniya


class StatsResponse(BaseModel):
    total_messages:    int
    total_users:       int
    total_cache_items: int
    messages_today:    int
    users_today:       int
    intent_breakdown:  dict   # {intent: count}
    top_queries:       list   # [{query, count}]


class UserListItem(BaseModel):
    user_key:      str
    message_count: int
    last_active:   str


class UserListResponse(BaseModel):
    items:   list[UserListItem]
    total:   int
    page:    int
    pages:   int


class MessageItem(BaseModel):
    role:       str
    content:    str
    created_at: str


class UserChatResponse(BaseModel):
    user_key: str
    items:    list[MessageItem]
    total:    int
    page:     int
    pages:    int


class CacheItem(BaseModel):
    id:         int
    query:      str
    answer:     str
    intent:     str
    created_at: str


class CacheListResponse(BaseModel):
    items:  list[CacheItem]
    total:  int
    page:   int
    pages:  int


# ─── ENDPOINTS ─────────────────────────────────────────────────────────────────

@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    """Admin login — token qaytaradi."""
    if body.username != ADMIN_USERNAME or body.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Login yoki parol noto'g'ri")

    token = _make_token(body.username)
    _VALID_TOKENS[token] = time.time() + TOKEN_TTL

    return LoginResponse(token=token, expires_in=TOKEN_TTL)


@router.get("/stats", response_model=StatsResponse)
async def get_stats(_: str = Depends(require_auth)):
    """Umumiy statistika."""
    pool = await get_pool()
    async with pool.acquire() as conn:

        # Jami xabarlar
        total_messages = await conn.fetchval(
            "SELECT COUNT(*) FROM chat_history"
        )

        # Jami foydalanuvchilar (unique user_key)
        total_users = await conn.fetchval(
            "SELECT COUNT(DISTINCT user_key) FROM chat_history"
        )

        # Jami cache
        total_cache = await conn.fetchval(
            "SELECT COUNT(*) FROM global_qa_cache"
        )

        # Bugungi xabarlar
        messages_today = await conn.fetchval(
            "SELECT COUNT(*) FROM chat_history WHERE created_at >= CURRENT_DATE"
        )

        # Bugungi yangi foydalanuvchilar
        users_today = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT user_key) FROM chat_history
            WHERE created_at >= CURRENT_DATE
            """
        )

        # Intent breakdown (global_qa_cache dan)
        intent_rows = await conn.fetch(
            """
            SELECT intent, COUNT(*) as cnt
            FROM global_qa_cache
            GROUP BY intent
            ORDER BY cnt DESC
            """
        )
        intent_breakdown = {r["intent"]: r["cnt"] for r in intent_rows}

        # Top 10 ko'p so'ralgan query (global_qa_cache)
        top_rows = await conn.fetch(
            """
            SELECT query, LENGTH(answer) as answer_len
            FROM global_qa_cache
            ORDER BY created_at DESC
            LIMIT 10
            """
        )
        top_queries = [
            {"query": r["query"][:80], "answer_len": r["answer_len"]}
            for r in top_rows
        ]

    return StatsResponse(
        total_messages=total_messages,
        total_users=total_users,
        total_cache_items=total_cache,
        messages_today=messages_today,
        users_today=users_today,
        intent_breakdown=intent_breakdown,
        top_queries=top_queries,
    )


@router.get("/chats", response_model=UserListResponse)
async def list_users(
    page:     int = 1,
    per_page: int = 20,
    search:   Optional[str] = None,
    _: str = Depends(require_auth),
):
    """
    Foydalanuvchilar ro'yxati — xabar soni va oxirgi faollik bilan.
    search: user_key bo'yicha filtr (ixtiyoriy)
    """
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:
        per_page = 20

    offset = (page - 1) * per_page
    pool   = await get_pool()

    async with pool.acquire() as conn:
        if search:
            where = "WHERE user_key ILIKE $1"
            args  = [f"%{search}%"]
        else:
            where = ""
            args  = []

        total = await conn.fetchval(
            f"SELECT COUNT(DISTINCT user_key) FROM chat_history {where}",
            *args,
        )

        rows = await conn.fetch(
            f"""
            SELECT
                user_key,
                COUNT(*)                          AS message_count,
                MAX(created_at)::TEXT             AS last_active
            FROM chat_history
            {where}
            GROUP BY user_key
            ORDER BY MAX(created_at) DESC
            LIMIT ${ len(args)+1 } OFFSET ${ len(args)+2 }
            """,
            *args, per_page, offset,
        )

    pages = max(1, (total + per_page - 1) // per_page)

    return UserListResponse(
        items=[
            UserListItem(
                user_key=r["user_key"],
                message_count=r["message_count"],
                last_active=r["last_active"],
            )
            for r in rows
        ],
        total=total,
        page=page,
        pages=pages,
    )


@router.get("/chats/{user_key}", response_model=UserChatResponse)
async def get_user_chat(
    user_key: str,
    page:     int = 1,
    per_page: int = 50,
    _: str = Depends(require_auth),
):
    """Bitta foydalanuvchining barcha yozishmalari (paginated, eski→yangi)."""
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 200:
        per_page = 50

    offset = (page - 1) * per_page
    pool   = await get_pool()

    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM chat_history WHERE user_key = $1",
            user_key,
        )

        rows = await conn.fetch(
            """
            SELECT role, content, created_at
            FROM chat_history
            WHERE user_key = $1
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
            """,
            user_key, per_page, offset,
        )

    if total == 0:
        raise HTTPException(status_code=404, detail="Foydalanuvchi topilmadi")

    pages = max(1, (total + per_page - 1) // per_page)

    return UserChatResponse(
        user_key=user_key,
        items=[
            MessageItem(
                role=r["role"],
                content=r["content"],
                created_at=r["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            )
            for r in rows
        ],
        total=total,
        page=page,
        pages=pages,
    )


@router.delete("/chats/{user_key}")
async def delete_user_chat(
    user_key: str,
    _: str = Depends(require_auth),
):
    """Bitta foydalanuvchining barcha chat tarixini o'chirish."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM chat_history WHERE user_key = $1",
            user_key,
        )
    count = int(result.split()[-1])
    if count == 0:
        raise HTTPException(
            status_code=404,
            detail="Foydalanuvchi topilmadi yoki tarix bo'sh",
        )
    return {"deleted": True, "user_key": user_key, "count": count}


@router.delete("/chats")
async def clear_all_chats(
    confirm: bool = False,
    _: str = Depends(require_auth),
):
    """
    BARCHA foydalanuvchilar chat tarixini o'chirish.
    Xavfli amal — ?confirm=true bo'lishi shart.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Barcha tarixni o'chirish uchun ?confirm=true qo'shing",
        )
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM chat_history")
    count = int(result.split()[-1])
    return {"deleted": True, "count": count, "scope": "all"}


@router.get("/cache", response_model=CacheListResponse)
async def list_cache(
    page:     int = 1,
    per_page: int = 20,
    intent:   Optional[str] = None,
    search:   Optional[str] = None,
    _: str = Depends(require_auth),
):
    """
    Global QA cache ro'yxati.
    intent: faq | location | offtopic (filtr)
    search: query matnida qidirish
    """
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:
        per_page = 20

    offset = (page - 1) * per_page
    pool   = await get_pool()

    conditions = []
    args: list = []

    if intent:
        args.append(intent)
        conditions.append(f"intent = ${len(args)}")

    if search:
        args.append(f"%{search}%")
        conditions.append(f"query ILIKE ${len(args)}")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    async with pool.acquire() as conn:
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM global_qa_cache {where}",
            *args,
        )

        rows = await conn.fetch(
            f"""
            SELECT id, query, answer, intent, created_at
            FROM global_qa_cache
            {where}
            ORDER BY created_at DESC
            LIMIT ${len(args)+1} OFFSET ${len(args)+2}
            """,
            *args, per_page, offset,
        )

    pages = max(1, (total + per_page - 1) // per_page)

    return CacheListResponse(
        items=[
            CacheItem(
                id=r["id"],
                query=r["query"],
                answer=r["answer"],
                intent=r["intent"],
                created_at=r["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
            )
            for r in rows
        ],
        total=total,
        page=page,
        pages=pages,
    )


@router.delete("/cache/{item_id}")
async def delete_cache_item(
    item_id: int,
    _: str = Depends(require_auth),
):
    """Global cache dan yozuvni o'chirish."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM global_qa_cache WHERE id = $1",
            item_id,
        )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Yozuv topilmadi")

    return {"deleted": True, "id": item_id}


@router.delete("/cache")
async def clear_cache_by_intent(
    intent: Optional[str] = None,
    _: str = Depends(require_auth),
):
    """
    Cache ni tozalash.
    intent berilsa — faqat o'sha intent o'chiriladi.
    Berilmasa — hammasi o'chiriladi.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        if intent:
            result = await conn.execute(
                "DELETE FROM global_qa_cache WHERE intent = $1",
                intent,
            )
        else:
            result = await conn.execute("DELETE FROM global_qa_cache")

    count = int(result.split()[-1])
    return {"deleted": True, "count": count, "intent": intent or "all"}


# ─── HEALTH ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok"}