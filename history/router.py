# history/router.py
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select, func, col

from history.database import get_session
from history.models import (
    ActionType,
    HistoryEntry,
    HistoryEntryCreate,
    HistoryEntryRead,
    HistoryListResponse,
)
from translation.nllb_engine import BURKINA_LANG_CODES
from tts.mms_engine import MMS_TTS_MODELS

router = APIRouter(prefix="/history", tags=["Historique"])

# ── Helpers de validation ─────────────────────────────────────────────

VALID_LANGS     = set(BURKINA_LANG_CODES.keys())
VALID_TTS_LANGS = set(MMS_TTS_MODELS.keys())


def validate_entry(entry: HistoryEntryCreate) -> None:
    """
    Applique les mêmes contraintes métier que /translate et /tts.
    Lève HTTPException 422 si une règle est violée.
    """
    errors = []

    # Contraintes selon action_type
    if entry.action_type in (ActionType.translate, ActionType.translate_and_speak):
        if not entry.src_lang:
            errors.append("src_lang requis pour translate / translate_and_speak")
        elif entry.src_lang not in VALID_LANGS:
            errors.append(f"src_lang inconnu : '{entry.src_lang}'. "
                          f"Valeurs acceptées : {sorted(VALID_LANGS)}")
        if not entry.tgt_lang:
            errors.append("tgt_lang requis pour translate / translate_and_speak")
        elif entry.tgt_lang not in VALID_LANGS:
            errors.append(f"tgt_lang inconnu : '{entry.tgt_lang}'. "
                          f"Valeurs acceptées : {sorted(VALID_LANGS)}")
        if not entry.source_text:
            errors.append("source_text requis pour translate / translate_and_speak")

    if entry.action_type in (ActionType.tts, ActionType.translate_and_speak):
        tts_lang = entry.tgt_lang if entry.action_type == ActionType.translate_and_speak \
                   else entry.src_lang
        if tts_lang and tts_lang not in VALID_TTS_LANGS:
            errors.append(f"Langue TTS non supportée : '{tts_lang}'. "
                          f"Valeurs acceptées : {sorted(VALID_TTS_LANGS)}")
        if not entry.source_text:
            errors.append("source_text requis pour tts")

    if entry.speed is not None and not (0.5 <= entry.speed <= 2.0):
        errors.append(f"speed doit être entre 0.5 et 2.0, reçu : {entry.speed}")

    if errors:
        raise HTTPException(status_code=422, detail=errors)


# ── POST /history ─────────────────────────────────────────────────────

@router.post(
    "",
    response_model=HistoryEntryRead,
    status_code=201,
    summary="Créer une entrée d'historique",
)
def create_history_entry(
    entry: HistoryEntryCreate,
    session: Session = Depends(get_session),
) -> HistoryEntryRead:
    """
    Crée une entrée d'historique après validation.
    - Valide les langues connues et les bornes de speed
    - Ne stocke jamais d'audio (régénéré à la demande via /tts)
    - Retourne 201 avec l'entrée créée
    """
    validate_entry(entry)

    db_entry = HistoryEntry(**entry.model_dump())
    session.add(db_entry)
    session.commit()
    session.refresh(db_entry)
    return db_entry


# ── GET /history ──────────────────────────────────────────────────────

@router.get(
    "",
    response_model=HistoryListResponse,
    summary="Lister l'historique (paginé)",
)
def list_history(
    user_id:     Optional[str]        = Query(None,  description="Filtrer par user_id"),
    action_type: Optional[ActionType] = Query(None,  description="Filtrer par type d'action"),
    lang:        Optional[str]        = Query(None,  description="Filtrer par src_lang OU tgt_lang"),
    limit:       int                  = Query(20,    ge=1, le=100, description="Nombre de résultats"),
    offset:      int                  = Query(0,     ge=0,         description="Décalage de pagination"),
    session:     Session              = Depends(get_session),
) -> HistoryListResponse:
    """
    Liste paginée de l'historique, triée par created_at desc.

    Filtres disponibles :
    - `user_id`     : filtre sur l'identifiant utilisateur
    - `action_type` : translate | tts | translate_and_speak
    - `lang`        : cherche dans src_lang ET tgt_lang
    - `limit`       : max 100 résultats par page
    - `offset`      : pagination
    """
    # ── Construction de la requête ────────────────────────────────────
    query       = select(HistoryEntry)
    count_query = select(func.count()).select_from(HistoryEntry)

    if user_id:
        query       = query.where(HistoryEntry.user_id == user_id)
        count_query = count_query.where(HistoryEntry.user_id == user_id)

    if action_type:
        query       = query.where(HistoryEntry.action_type == action_type)
        count_query = count_query.where(HistoryEntry.action_type == action_type)

    if lang:
        lang_filter = (
            (col(HistoryEntry.src_lang) == lang) |
            (col(HistoryEntry.tgt_lang) == lang)
        )
        query       = query.where(lang_filter)
        count_query = count_query.where(lang_filter)

    # ── Tri + pagination ──────────────────────────────────────────────
    query = query.order_by(col(HistoryEntry.created_at).desc())
    query = query.offset(offset).limit(limit)

    total = session.exec(count_query).one()
    items = session.exec(query).all()

    return HistoryListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=items,
    )


# ── DELETE /history/{id} ──────────────────────────────────────────────

@router.delete(
    "/{entry_id}",
    status_code=204,
    summary="Supprimer une entrée",
)
def delete_history_entry(
    entry_id: uuid.UUID,
    session:  Session = Depends(get_session),
) -> None:
    """
    Supprime une entrée par son UUID.
    - 404 si non trouvée
    - 204 si supprimée avec succès
    """
    entry = session.get(HistoryEntry, entry_id)
    if not entry:
        raise HTTPException(
            status_code=404,
            detail=f"Entrée introuvable : {entry_id}",
        )
    session.delete(entry)
    session.commit()


# ── DELETE /history ───────────────────────────────────────────────────

@router.delete(
    "",
    status_code=204,
    summary="Vider tout l'historique de l'utilisateur",
)
def clear_history(
    user_id: str     = Query(..., description="user_id dont on vide l'historique"),
    session: Session = Depends(get_session),
) -> None:
    """
    Supprime toutes les entrées d'un utilisateur.
    - user_id obligatoire pour éviter de vider tout l'historique global par erreur
    - 204 dans tous les cas (idempotent)
    """
    entries = session.exec(
        select(HistoryEntry).where(HistoryEntry.user_id == user_id)
    ).all()

    for entry in entries:
        session.delete(entry)
    session.commit()