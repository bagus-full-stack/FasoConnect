# api/main.py — VERSION FINALE COMPLÈTE
import os
import json
import hashlib
import logging
import asyncio
import time
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException, Request, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from huggingface_hub import login
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from translation.nllb_engine import NLLBTranslator, BURKINA_LANG_CODES
from tts.mms_engine import MMSTTSEngine, MMS_TTS_MODELS
from history.database import create_db_tables, get_session
from history.router import router as history_router

# ── Logging JSON structuré ────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": self.formatTime(record),
            "level":     record.levelname,
            "message":   record.getMessage(),
        }
        if hasattr(record, "extra"):
            log.update(record.extra)
        return json.dumps(log, ensure_ascii=False)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

API_KEY        = os.getenv("API_KEY")
HF_TOKEN       = os.getenv("HF_TOKEN")
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
CACHE_TTL      = int(os.getenv("CACHE_TTL", "3600"))
INFER_TIMEOUT  = int(os.getenv("INFER_TIMEOUT", "60"))

# ── Auth API Key ──────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    if not API_KEY:
        return True
    if key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Clé API invalide ou manquante. Ajoute le header X-API-Key."
        )
    return True

# ── Rate Limiting ─────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Cache Redis ───────────────────────────────────────────────────────

CACHE_ENABLED = False
cache = None

try:
    cache = redis.Redis(host="redis", port=6379, decode_responses=True)
    cache.ping()
    CACHE_ENABLED = True
    logger.info("Redis connecté", extra={"event": "redis_connected"})
except Exception:
    logger.warning("Redis non disponible — cache désactivé",
                   extra={"event": "redis_unavailable"})

def cache_get(key: str):
    if not CACHE_ENABLED:
        return None
    try:
        value = cache.get(key)
        return json.loads(value) if value else None
    except Exception:
        return None

def cache_set(key: str, value: dict):
    if not CACHE_ENABLED:
        return
    try:
        cache.setex(key, CACHE_TTL, json.dumps(value))
    except Exception:
        pass

def make_cache_key(prefix: str, *args) -> str:
    raw = ":".join(str(a) for a in args)
    return f"{prefix}:{hashlib.md5(raw.encode()).hexdigest()}"

# ── Lifespan ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Auth HuggingFace
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
        logger.info("HuggingFace authentifié", extra={"event": "hf_auth_ok"})
    else:
        logger.warning("HF_TOKEN manquant", extra={"event": "hf_auth_missing"})

    # 2. Base de données
    create_db_tables()
    logger.info("Base de données prête", extra={"event": "db_ready"})

    # 3. NLLB-200
    logger.info("Chargement NLLB-200...", extra={"event": "model_loading", "model": "nllb"})
    t0 = time.time()
    app.state.translator = NLLBTranslator()
    logger.info("NLLB-200 prêt", extra={
        "event": "model_ready", "model": "nllb",
        "device": str(app.state.translator.device),
        "duration_s": round(time.time() - t0, 1),
    })

    # 4. MMS-TTS
    logger.info("Chargement MMS-TTS...", extra={"event": "model_loading", "model": "mms"})
    t0 = time.time()
    app.state.tts = MMSTTSEngine()
    logger.info("MMS-TTS prêt", extra={
        "event": "model_ready", "model": "mms",
        "duration_s": round(time.time() - t0, 1),
    })

    yield

    logger.info("Arrêt de l'API", extra={"event": "shutdown"})

# ── Application ───────────────────────────────────────────────────────

app = FastAPI(
    title="FasoConnect — API Linguistique Burkinabè",
    description=(
        "Traduction automatique et synthèse vocale pour les langues du Burkina Faso.\n\n"
        "**Langues supportées :** Mooré · Dioula · Fulfulde · Gourmantchéma · Dagaare · Français · Anglais\n\n"
        "**Modèles :** Meta NLLB-200 distilled 600M + Meta MMS-TTS\n\n"
        "**Auth :** header `X-API-Key` requis si `API_KEY` configurée dans `.env`"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middlewares ───────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Trop de requêtes. Réessaie dans quelques secondes."},
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code}",
        extra={
            "event":       "http_request",
            "method":      request.method,
            "path":        request.url.path,
            "status":      response.status_code,
            "duration_ms": round((time.time() - t0) * 1000),
            "ip":          request.client.host if request.client else "unknown",
        }
    )
    return response

# ── Router historique ─────────────────────────────────────────────────

app.include_router(history_router)

# ── Schémas ───────────────────────────────────────────────────────────

class TranslationRequest(BaseModel):
    text:     str   = Field(..., min_length=1, max_length=2000, example="Bonjour tout le monde")
    src_lang: str   = Field(..., example="francais")
    tgt_lang: str   = Field(..., example="moore")

class TranslationResponse(BaseModel):
    translated_text: str
    src_lang:        str
    tgt_lang:        str
    cached:          bool = False

class TTSRequest(BaseModel):
    text:  str   = Field(..., min_length=1, max_length=500, example="Ne y welame")
    lang:  str   = Field(..., example="moore")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

class TTSResponse(BaseModel):
    audio_b64:        str
    sample_rate:      int
    duration_seconds: float
    lang:             str

class TranslateAndSpeakRequest(BaseModel):
    text:     str   = Field(..., min_length=1, max_length=2000)
    src_lang: str
    tgt_lang: str
    speed:    float = Field(default=1.0, ge=0.5, le=2.0)

# ── Helper inférence async + timeout ─────────────────────────────────

async def run_with_timeout(fn, *args, timeout: int = None):
    """
    Exécute une inférence synchrone dans un thread séparé
    sans bloquer l'event loop FastAPI.
    Lève 504 si le timeout est dépassé.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args),
            timeout=timeout or INFER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Inférence trop longue (>{timeout or INFER_TIMEOUT}s). "
                   "Réessaie avec un texte plus court."
        )

# ── Endpoints système ─────────────────────────────────────────────────

@app.get(
    "/health",
    tags=["Système"],
    summary="Healthcheck",
)
async def health():
    """Vérifie que l'API est opérationnelle — utilisé par Docker et les load balancers."""
    return {
        "status":  "ok",
        "version": "1.0.0",
        "models":  ["nllb-200-distilled-600M", "mms-tts"],
        "cache":   "redis" if CACHE_ENABLED else "disabled",
    }

@app.get(
    "/languages",
    tags=["Système"],
    summary="Langues disponibles",
)
async def list_languages():
    """Retourne les langues supportées pour la traduction et la synthèse vocale."""
    return {
        "translation_supported": list(BURKINA_LANG_CODES.keys()),
        "tts_supported":         list(MMS_TTS_MODELS.keys()),
        "nllb_codes":            BURKINA_LANG_CODES,
    }

# ── Traduction ────────────────────────────────────────────────────────

@app.post(
    "/translate",
    response_model=TranslationResponse,
    status_code=200,
    tags=["Traduction"],
    summary="Traduire un texte",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("20/minute")
async def translate(request: Request, req: TranslationRequest):
    """
    Traduit un texte entre deux langues supportées.

    - **Rate limit** : 20 requêtes/minute par IP
    - **Cache** : résultat mis en cache Redis 1h
    - **Async** : ne bloque pas les autres requêtes pendant l'inférence
    """
    cache_key = make_cache_key("trans", req.text, req.src_lang, req.tgt_lang)
    cached = cache_get(cache_key)
    if cached:
        return TranslationResponse(**cached, cached=True)

    try:
        result = await run_with_timeout(
            request.app.state.translator.translate,
            req.text, req.src_lang, req.tgt_lang,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur traduction : {e}", extra={"event": "translation_error"})
        raise HTTPException(status_code=500, detail="Erreur interne lors de la traduction")

    data = {
        "translated_text": result,
        "src_lang":        req.src_lang,
        "tgt_lang":        req.tgt_lang,
    }
    cache_set(cache_key, data)
    return TranslationResponse(**data, cached=False)

# ── Synthèse vocale ───────────────────────────────────────────────────

@app.post(
    "/tts",
    response_model=TTSResponse,
    status_code=200,
    tags=["Synthèse vocale"],
    summary="Synthétiser un texte en audio",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def text_to_speech(request: Request, req: TTSRequest):
    """
    Génère un audio WAV encodé en base64 à partir d'un texte.

    - **Rate limit** : 10 requêtes/minute par IP
    - **Speed** : 0.5 (lent) → 1.0 (normal) → 2.0 (rapide)
    - **Format** : WAV 16000 Hz, mono, encodé base64
    """
    cache_key = make_cache_key("tts", req.text, req.lang, req.speed)
    cached = cache_get(cache_key)
    if cached:
        return TTSResponse(**cached)

    try:
        result = await run_with_timeout(
            request.app.state.tts.synthesize,
            req.text, req.lang, req.speed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur TTS : {e}", extra={"event": "tts_error"})
        raise HTTPException(status_code=500, detail="Erreur interne lors de la synthèse vocale")

    data = {
        "audio_b64":        result.audio_b64,
        "sample_rate":      result.sample_rate,
        "duration_seconds": result.duration_seconds,
        "lang":             req.lang,
    }
    cache_set(cache_key, data)
    return TTSResponse(**data)

# ── Pipeline ──────────────────────────────────────────────────────────

@app.post(
    "/translate-and-speak",
    status_code=200,
    tags=["Pipeline"],
    summary="Traduire puis synthétiser en audio",
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def translate_and_speak(request: Request, req: TranslateAndSpeakRequest):
    """
    Pipeline combiné : traduction + synthèse vocale en une seule requête.

    - **Rate limit** : 10 requêtes/minute par IP
    - Retourne le texte traduit ET l'audio base64
    - Résultat mis en cache Redis 1h
    """
    cache_key = make_cache_key("pipeline", req.text, req.src_lang, req.tgt_lang, req.speed)
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    # Étape 1 — Traduction
    try:
        translated = await run_with_timeout(
            request.app.state.translator.translate,
            req.text, req.src_lang, req.tgt_lang,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur traduction : {e}")

    # Étape 2 — Synthèse vocale sur le texte traduit
    try:
        audio = await run_with_timeout(
            request.app.state.tts.synthesize,
            translated, req.tgt_lang, req.speed,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur synthèse vocale : {e}")

    data = {
        "original_text":    req.text,
        "translated_text":  translated,
        "src_lang":         req.src_lang,
        "tgt_lang":         req.tgt_lang,
        "audio_b64":        audio.audio_b64,
        "sample_rate":      audio.sample_rate,
        "duration_seconds": audio.duration_seconds,
        "cached":           False,
    }
    cache_set(cache_key, data)
    return data