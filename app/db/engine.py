import os
from collections.abc import AsyncGenerator
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing")

"""
NOTE: don't forget to add the database connection on startup
"""

# engine manages underlying database connection
# we use async driver, because sync drivers like psycopg2 blocks
# the thread on every query. While asyncpg waits non blockingly 
# and keeps the event loop free. Especially with our API supporting
# concurrent operations
engine = create_async_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=3
    )

# Factory producing one AsyncSession per use. expire_on_commit=False keeps
# attributes usable after background commit without lazy reloads.
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    # Per-request session(transaction): lends one pooled connection, guarantees close.
    async with async_session() as session:
        yield session