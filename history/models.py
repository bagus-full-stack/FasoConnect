# history/models.py
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class ActionType(str, Enum):
    translate           = "translate"
    tts                 = "tts"
    translate_and_speak = "translate_and_speak"


class HistoryEntry(SQLModel, table=True):
    """
    Entrée d'historique — une ligne par appel à /translate, /tts ou /translate-and-speak.
    L'audio n'est jamais stocké : il est régénéré à la demande via /tts.
    """
    __tablename__ = "history"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
    )
    user_id: Optional[str] = Field(
        default=None,
        index=True,
        max_length=128,
        description="Identifiant utilisateur ou session anonyme",
    )
    action_type: ActionType = Field(
        index=True,
        description="Type d'action effectuée",
    )
    src_lang: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Langue source (nullable pour TTS)",
    )
    tgt_lang: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Langue cible (nullable pour TTS)",
    )
    source_text: Optional[str] = Field(
        default=None,
        description="Texte soumis par l'utilisateur",
    )
    result_text: Optional[str] = Field(
        default=None,
        description="Texte traduit ou texte synthétisé (sans audio)",
    )
    speed: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=2.0,
        description="Vitesse TTS (0.5 → 2.0)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        index=True,
        description="Date de création (UTC)",
    )


# ── Schémas Pydantic (lecture/écriture API) ───────────────────────────

class HistoryEntryRead(SQLModel):
    """Schéma retourné par l'API."""
    id:          uuid.UUID
    user_id:     Optional[str]
    action_type: ActionType
    src_lang:    Optional[str]
    tgt_lang:    Optional[str]
    source_text: Optional[str]
    result_text: Optional[str]
    speed:       Optional[float]
    created_at:  datetime


class HistoryEntryCreate(SQLModel):
    """Schéma de création — validé avant insertion."""
    user_id:     Optional[str]    = None
    action_type: ActionType
    src_lang:    Optional[str]    = None
    tgt_lang:    Optional[str]    = None
    source_text: Optional[str]    = Field(default=None, max_length=2000)
    result_text: Optional[str]    = Field(default=None, max_length=2000)
    speed:       Optional[float]  = Field(default=None, ge=0.5, le=2.0)


class HistoryListResponse(SQLModel):
    """Réponse paginée pour GET /history."""
    total:   int
    limit:   int
    offset:  int
    items:   list[HistoryEntryRead]