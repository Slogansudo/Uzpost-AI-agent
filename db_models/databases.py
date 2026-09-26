from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.sql import func
import uuid
import os
from dotenv import load_dotenv
load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")

# Engine yaratish
engine = create_engine(
    DATABASE_URL,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# engine = create_engine(DATABASE_URL, echo=False)
# SessionLocal = sessionmaker(bind=engine)
#Base = declarative_base()


# ========================
# 1. user_new_chat_sessions
# ========================
class UserNewChatSession(Base):
    __tablename__ = "user_new_chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_key = Column(Text, nullable=False)

    session_id = Column(
        String(36),
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True
    )
    title = Column(Text, default="Yangi suhbat")
    created_at = Column(DateTime, server_default=func.now())
    last_active = Column(DateTime, server_default=func.now())


# ========================
# 2. user_chat_messages
# ========================
class UserChatMessage(Base):
    __tablename__ = "user_chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(
        String(36),
        ForeignKey("user_new_chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="role_check"),
    )

    # Relationship (child → parent)
    session = relationship("UserNewChatSession", backref="messages")


# ========================
# 3. user_sessions
# ========================
class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)  # unique=True olib tashlandi (cache uchun)
    answer = Column(Text, nullable=False)

    user_key = Column(String(255), index=True)
    client_ip = Column(String(50))
    user_agent = Column(Text)

    hit_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())



# ========================
# 4. global_qa_cache
# ========================
class GlobalQACache(Base):
    __tablename__ = "global_qa_cache"

    id         = Column(Integer, primary_key=True)
    query      = Column(Text, nullable=False, unique=True)   # unique — takrorlanmaydi
    answer     = Column(Text, nullable=False)
    intent     = Column(String(20), nullable=False)          # faq | location | offtopic
    created_at = Column(DateTime, server_default=func.now())


# ========================
# DB INIT
# ========================
def init_db():
    Base.metadata.create_all(engine)
    print("✅ Database initialized: uzpost_assistant.db")


if __name__ == "__main__":
    init_db()
