# api/main.py
import os
import json
import hashlib
import logging
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import login

from translation.nllb_engine import NLLBTranslator, BURKINA_LANG_CODES
from tts.mms_engine import MMSTTSEngine, MMS_TTS_MODELS

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Cache Redis ───────────────────────────────────────────────────────

CACHE_TTL = 3600  # 1 heure

try:
    cache = redis.Redis(host="redis", port=6379, decode_responses=True)
    cache.ping()
    CACHE_ENABLED = True
    logging.info("✅ Redis connecté")
except Exception:
    cache = None
    CACHE_ENABLED = False
    logging.warning("⚠️  Redis non disponible — cache désactivé")


# ── Lifespan : startup / shutdown ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Authentification HuggingFace
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
        logging.info("✅ HuggingFace authentifié")
    else:
        logging.warning("⚠️  HF_TOKEN manquant — requêtes non authentifiées")

    # 2. Chargement des modèles
    logging.info("⏳ Chargement NLLB-200...")
    app.state.translator = NLLBTranslator()
    logging.info("✅ NLLB-200 prêt")

    logging.info("⏳ Chargement MMS-TTS...")
    app.state.tts = MMSTTSEngine()
    logging.info("✅ MMS-TTS prêt")

    yield  # ← l'API accepte les requêtes à partir d'ici

    # Shutdown
    logging.info("🛑 Arrêt de l'API")


# ── Application ───────────────────────────────────────────────────────

app = FastAPI(
    title="API Linguistique Burkinabè — FasoConnect",
    description="Traduction & synthèse vocale pour les langues du Burkina Faso (NLLB-200 + MMS-TTS)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schémas Pydantic ──────────────────────────────────────────────────

class TranslationRequest(BaseModel):
    text: str = Field(..., max_length=2000, example="Bonjour tout le monde")
    src_lang: str = Field(..., example="francais")
    tgt_lang: str = Field(..., example="moore")


class TranslationResponse(BaseModel):
    translated_text: str
    src_lang: str
    tgt_lang: str
    cached: bool = False


class TTSRequest(BaseModel):
    text: str = Field(..., max_length=500, example="Bonjour")
    lang: str = Field(..., example="moore")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class TTSResponse(BaseModel):
    audio_b64: str
    sample_rate: int
    duration_seconds: float
    lang: str


class TranslateAndSpeakRequest(BaseModel):
    text: str = Field(..., max_length=2000, example="Bonjour tout le monde")
    src_lang: str = Field(..., example="francais")
    tgt_lang: str = Field(..., example="moore")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


# ── Helpers ───────────────────────────────────────────────────────────

def make_cache_key(prefix: str, *args) -> str:
    raw = ":".join(str(a) for a in args)
    return f"{prefix}:{hashlib.md5(raw.encode()).hexdigest()}"


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


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", tags=["Système"])
async def health():
    return {
        "status": "ok",
        "models": ["nllb-200", "mms-tts"],
        "cache": "redis" if CACHE_ENABLED else "disabled",
    }


@app.get("/languages", tags=["Système"])
async def list_languages():
    """Retourne les langues supportées pour la traduction et la synthèse vocale."""
    return {
        "translation_supported": list(BURKINA_LANG_CODES.keys()),
        "tts_supported": list(MMS_TTS_MODELS.keys()),
        "nllb_codes": BURKINA_LANG_CODES,
    }


@app.post("/translate", response_model=TranslationResponse, tags=["Traduction"])
async def translate(req: TranslationRequest, request: Request):
    """Traduit un texte entre deux langues supportées."""
    cache_key = make_cache_key("trans", req.text, req.src_lang, req.tgt_lang)
    cached = cache_get(cache_key)
    if cached:
        return TranslationResponse(**cached, cached=True)

    translator = request.app.state.translator
    try:
        result = translator.translate(req.text, req.src_lang, req.tgt_lang)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Erreur traduction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la traduction")

    data = {
        "translated_text": result,
        "src_lang": req.src_lang,
        "tgt_lang": req.tgt_lang,
    }
    cache_set(cache_key, data)
    return TranslationResponse(**data, cached=False)


@app.post("/tts", response_model=TTSResponse, tags=["Synthèse vocale"])
async def text_to_speech(req: TTSRequest, request: Request):
    """Génère un fichier audio WAV (base64) à partir d'un texte."""
    cache_key = make_cache_key("tts", req.text, req.lang, req.speed)
    cached = cache_get(cache_key)
    if cached:
        return TTSResponse(**cached)

    tts = request.app.state.tts
    try:
        result = tts.synthesize(req.text, req.lang, req.speed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Erreur TTS : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la synthèse vocale")

    data = {
        "audio_b64": result.audio_b64,
        "sample_rate": result.sample_rate,
        "duration_seconds": result.duration_seconds,
        "lang": req.lang,
    }
    cache_set(cache_key, data)
    return TTSResponse(**data)


@app.post("/translate-and-speak", tags=["Pipeline"])
async def translate_and_speak(req: TranslateAndSpeakRequest, request: Request):
    """Pipeline combiné : traduction puis synthèse vocale en une seule requête."""
    cache_key = make_cache_key("pipeline", req.text, req.src_lang, req.tgt_lang, req.speed)
    cached = cache_get(cache_key)
    if cached:
        return {**cached, "cached": True}

    translator = request.app.state.translator
    tts = request.app.state.tts

    try:
        translated = translator.translate(req.text, req.src_lang, req.tgt_lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur traduction : {e}")

    try:
        audio = tts.synthesize(translated, req.tgt_lang, req.speed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur synthèse vocale : {e}")

    data = {
        "original_text": req.text,
        "translated_text": translated,
        "src_lang": req.src_lang,
        "tgt_lang": req.tgt_lang,
        "audio_b64": audio.audio_b64,
        "sample_rate": audio.sample_rate,
        "duration_seconds": audio.duration_seconds,
        "cached": False,
    }
    cache_set(cache_key, data)
    return data