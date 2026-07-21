# history/database.py
import os
from sqlmodel import SQLModel, create_engine, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fasoconnect.db")

# ── Engine ────────────────────────────────────────────────────────────
# SQLite pour le dev, PostgreSQL pour la prod :
# DATABASE_URL=postgresql://user:pass@localhost:5432/fasoconnect

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,           # True pour voir les requêtes SQL dans les logs
    connect_args=connect_args,
)


def create_db_tables():
    """Crée toutes les tables si elles n'existent pas encore."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Générateur de session SQLModel — utilisé via Depends() dans FastAPI."""
    with Session(engine) as session:
        yield session