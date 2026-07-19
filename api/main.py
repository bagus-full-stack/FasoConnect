# api/main.py — VERSION PRODUCTION
import os
import json
import hashlib
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

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

# ── Configuration depuis .env ─────────────────────────────────────────

API_KEY        = os.getenv("API_KEY")               # clé d'accès à l'API
HF_TOKEN       = os.getenv("HF_TOKEN")              # token HuggingFace
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")   # ex: https://ton-domaine.com
CACHE_TTL      = int(os.getenv("CACHE_TTL", "3600"))
INFER_TIMEOUT  = int(os.getenv("INFER_TIMEOUT", "60"))  # secondes

# ── Authentification API Key ─────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    """Vérifie la clé API si API_KEY est définie dans .env."""
    if not API_KEY:
        # Pas de clé configurée → mode ouvert (dev uniquement)
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
    logger.warning("Redis non disponible — cache désactivé", extra={"event": "redis_unavailable"})

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

    # 2. Chargement NLLB-200
    logger.info("Chargement NLLB-200...", extra={"event": "model_loading", "model": "nllb"})
    t0 = time.time()
    app.state.translator = NLLBTranslator()
    logger.info("NLLB-200 prêt", extra={"event": "model_ready", "model": "nllb",
                                         "duration_s": round(time.time() - t0, 1)})

    # 3. Chargement MMS-TTS
    logger.info("Chargement MMS-TTS...", extra={"event": "model_loading", "model": "mms"})
    t0 = time.time()
    app.state.tts = MMSTTSEngine()
    logger.info("MMS-TTS prêt", extra={"event": "model_ready", "model": "mms",
                                        "duration_s": round(time.time() - t0, 1)})

    yield

    logger.info("Arrêt de l'API", extra={"event": "shutdown"})

# ── Application ───────────────────────────────────────────────────────

app = FastAPI(
    title="API Linguistique Burkinabè — FasoConnect",
    description="Traduction & synthèse vocale pour les langues du Burkina Faso",
    version="1.0.0",
    lifespan=lifespan,
    # Désactive la doc en prod si souhaité
    # docs_url=None, redoc_url=None,
)

# ── Middleware CORS ───────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],   # ← restreint via .env
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ── Rate Limiter middleware ───────────────────────────────────────────

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Trop de requêtes — réessaie dans quelques secondes."},
    )

# ── Middleware de logging des requêtes ────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - t0) * 1000)
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code}",
        extra={
            "event":       "http_request",
            "method":      request.method,
            "path":        request.url.path,
            "status":      response.status_code,
            "duration_ms": duration_ms,
            "ip":          request.client.host,
        }
    )
    return response

# ── Schémas Pydantic ──────────────────────────────────────────────────

class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, example="Bonjour tout le monde")
    src_lang: str = Field(..., example="francais")
    tgt_lang: str = Field(..., example="moore")

class TranslationResponse(BaseModel):
    translated_text: str
    src_lang: str
    tgt_lang: str
    cached: bool = False

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, example="Bonjour")
    lang: str = Field(..., example="moore")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

class TTSResponse(BaseModel):
    audio_b64: str
    sample_rate: int
    duration_seconds: float
    lang: str

class TranslateAndSpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    src_lang: str
    tgt_lang: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)

# ── Helper : inférence async avec timeout ─────────────────────────────

async def run_with_timeout(fn, *args, timeout: int = None):
    """
    Exécute une fonction synchrone (inférence modèle) dans un thread séparé
    pour ne pas bloquer l'event loop FastAPI.
    Lève une HTTPException 504 si le timeout est dépassé.
    """
    _timeout = timeout or INFER_TIMEOUT
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(fn, *args),
            timeout=_timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Inférence trop longue (timeout {_timeout}s). Réessaie avec un texte plus court."
        )

# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", tags=["Système"])
async def health():
    """Healthcheck — utilisé par Docker et les load balancers."""
    return {
        "status":  "ok",
        "models":  ["nllb-200", "mms-tts"],
        "cache":   "redis" if CACHE_ENABLED else "disabled",
        "version": "1.0.0",
    }

@app.get("/languages", tags=["Système"])
async def list_languages():
    """Retourne les langues supportées pour la traduction et la synthèse vocale."""
    return {
        "translation_supported": list(BURKINA_LANG_CODES.keys()),
        "tts_supported":         list(MMS_TTS_MODELS.keys()),
        "nllb_codes":            BURKINA_LANG_CODES,
    }

@app.post(
    "/translate",
    response_model=TranslationResponse,
    tags=["Traduction"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("20/minute")
async def translate(request: Request, req: TranslationRequest):
    """
    Traduit un texte entre deux langues supportées.
    - Rate limit : 20 requêtes/minute par IP
    - Auth : header X-API-Key requis si API_KEY définie dans .env
    """
    cache_key = make_cache_key("trans", req.text, req.src_lang, req.tgt_lang)
    cached = cache_get(cache_key)
    if cached:
        return TranslationResponse(**cached, cached=True)

    translator = request.app.state.translator

    try:
        # ✅ Async — ne bloque pas les autres requêtes
        result = await run_with_timeout(
            translator.translate,
            req.text, req.src_lang, req.tgt_lang,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur traduction : {e}", extra={"event": "translation_error", "error": str(e)})
        raise HTTPException(status_code=500, detail="Erreur interne lors de la traduction")

    data = {"translated_text": result, "src_lang": req.src_lang, "tgt_lang": req.tgt_lang}
    cache_set(cache_key, data)
    return TranslationResponse(**data, cached=False)

@app.post(
    "/tts",
    response_model=TTSResponse,
    tags=["Synthèse vocale"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def text_to_speech(request: Request, req: TTSRequest):
    """
    Génère un audio WAV encodé en base64 à partir d'un texte.
    - Rate limit : 10 requêtes/minute par IP
    """
    cache_key = make_cache_key("tts", req.text, req.lang, req.speed)
    cached = cache_get(cache_key)
    if cached:
        return TTSResponse(**cached)

    tts = request.app.state.tts

    try:
        result = await run_with_timeout(
            tts.synthesize,
            req.text, req.lang, req.speed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur TTS : {e}", extra={"event": "tts_error", "error": str(e)})
        raise HTTPException(status_code=500, detail="Erreur interne lors de la synthèse vocale")

    data = {
        "audio_b64":        result.audio_b64,
        "sample_rate":      result.sample_rate,
        "duration_seconds": result.duration_seconds,
        "lang":             req.lang,
    }
    cache_set(cache_key, data)
    return TTSResponse(**data)

@app.post(
    "/translate-and-speak",
    tags=["Pipeline"],
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("10/minute")
async def translate_and_speak(request: Request, req: TranslateAndSpeakRequest):
    """
    Pipeline combiné : traduction + synthèse vocale en une seule requête.
    - Rate limit : 10 requêtes/minute par IP
    """
    cache_key = make_cache_key("pipeline", req.text, req.src_lang, req.tgt_lang, req.speed)
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    translator = request.app.state.translator
    tts        = request.app.state.tts

    try:
        translated = await run_with_timeout(
            translator.translate,
            req.text, req.src_lang, req.tgt_lang,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur traduction : {e}")

    try:
        audio = await run_with_timeout(
            tts.synthesize,
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