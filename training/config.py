# training/config.py — Configuration centralisée du fine-tuning
"""
Tous les hyperparamètres en un seul endroit.
Modifie ce fichier plutôt que finetune_nllb.py directement.
"""
from pathlib import Path

# ── Modèle ────────────────────────────────────────────────────────────

# Modèle de base à fine-tuner
BASE_MODEL = "facebook/nllb-200-distilled-600M"

# Dossier de sortie du modèle fine-tuné
OUTPUT_DIR = "models/nllb-burkina-v1"

# Dossier des checkpoints intermédiaires
CHECKPOINT_DIR = "models/checkpoints"

# ── Données ───────────────────────────────────────────────────────────

CORPUS_FILE       = "data/processed/corpus_burkina_clean.csv"
RADIO_CORPUS_FILE = "data/processed/corpus_burkina_radio.csv"
LOG_DIR           = "logs/tensorboard"

# ── Langues ───────────────────────────────────────────────────────────

BURKINA_LANGS = [
    ("moore",        "mos", "mos_Latn"),
    ("dioula",       "dyu", "dyu_Latn"),
    ("fulfulde",     "fuv", "fuv_Latn"),
    ("gourmantsema", "gux", "gux_Latn"),
    ("dagaare",      "dga", "dga_Latn"),
]

PIVOT_LANGS = [
    ("francais", "fra", "fra_Latn"),
    ("anglais",  "eng", "eng_Latn"),
]

# Mapping complet code interne → code NLLB
LANG_CODES = {
    "moore":        "mos_Latn",
    "dioula":       "dyu_Latn",
    "fulfulde":     "fuv_Latn",
    "gourmantsema": "gux_Latn",
    "dagaare":      "dga_Latn",
    "francais":     "fra_Latn",
    "anglais":      "eng_Latn",
    # codes ISO courts
    "mos": "mos_Latn",
    "dyu": "dyu_Latn",
    "fuv": "fuv_Latn",
    "gux": "gux_Latn",
    "dga": "dga_Latn",
    "fra": "fra_Latn",
    "eng": "eng_Latn",
}

# ── Hyperparamètres d'entraînement ────────────────────────────────────

TRAIN_EPOCHS          = 5
BATCH_SIZE            = 8       # réduis à 4 si OOM sur GPU 6 Go
EVAL_BATCH_SIZE       = 8
LEARNING_RATE         = 5e-5
WARMUP_STEPS          = 500
WEIGHT_DECAY          = 0.01
LR_SCHEDULER          = "cosine"
EARLY_STOP_PATIENCE   = 2
SAVE_TOTAL_LIMIT      = 2       # garde les N meilleurs checkpoints

# ── Tokenisation ──────────────────────────────────────────────────────

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256

# ── Nettoyage du corpus ───────────────────────────────────────────────

MIN_CHARS           = 5        # longueur minimale source et cible
MAX_CHARS           = 1000     # longueur maximale
MIN_LASER_SCORE     = 0.6      # filtre qualité allenai/nllb
DEDUP               = True     # suppression des doublons
INCLUDE_RADIO       = True     # inclure corpus radio si disponible
EXCLUDE_NEEDS_REVIEW = True    # exclure paires radio non validées

# ── Évaluation ────────────────────────────────────────────────────────

EVAL_DATASET   = "facebook/flores"    # dataset d'évaluation finale
EVAL_SPLIT     = "devtest"            # split FLORES-200
EVAL_LANG_PAIR = ("moore", "francais")  # paire à évaluer en priorité

# ── Whisper (transcription radio) ─────────────────────────────────────

WHISPER_MODEL_SIZE = "large-v3"   # tiny | base | small | medium | large-v3
WHISPER_BOOTSTRAP  = True         # traduit avec NLLB après transcription

# ── Dossiers à créer automatiquement ─────────────────────────────────

REQUIRED_DIRS = [
    "data/raw/audio",
    "data/raw/opus",
    "data/raw/transcriptions",
    "data/processed",
    "models",
    "logs/tensorboard",
]