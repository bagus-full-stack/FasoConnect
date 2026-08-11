# training/config.py — VERSION FINALE
"""
Configuration centralisée du pipeline de fine-tuning.
Toutes les constantes sont définies ici.
Importé par finetune_nllb.py et evaluate_model.py.
"""
from pathlib import Path

# ── Modèle ────────────────────────────────────────────────────────────

BASE_MODEL     = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR     = "models/nllb-burkina-v1"
CHECKPOINT_DIR = "models/checkpoints"

# ── Données ───────────────────────────────────────────────────────────

# ✅ Pointe vers le corpus NETTOYÉ (produit par clean_corpus.py)
CORPUS_FILE       = "data/processed/corpus_burkina_clean.csv"
RADIO_CORPUS_FILE = "data/processed/corpus_burkina_radio.csv"
LOG_DIR           = "logs/tensorboard"

# ── Langues ───────────────────────────────────────────────────────────

BURKINA_LANGS = [
    ("moore",        "mos", "mos_Latn"),
    ("dioula",       "dyu", "dyu_Latn"),
    ("bambara",      "bam", "bam_Latn"),
    ("fulfulde",     "fuv", "fuv_Latn"),
    ("gourmantsema", "gux", "gux_Latn"),
    ("dagaare",      "dga", "dga_Latn"),
]

PIVOT_LANGS = [
    ("francais", "fra", "fra_Latn"),
    ("anglais",  "eng", "eng_Latn"),
]

LANG_CODES = {
    # codes internes
    "moore":        "mos_Latn",
    "dioula":       "dyu_Latn",
    "bambara":      "bam_Latn",
    "fulfulde":     "fuv_Latn",
    "gourmantsema": "gux_Latn",
    "dagaare":      "dga_Latn",
    "francais":     "fra_Latn",
    "anglais":      "eng_Latn",
    # codes ISO courts
    "mos": "mos_Latn",
    "dyu": "dyu_Latn",
    "bam": "bam_Latn",
    "fuv": "fuv_Latn",
    "gux": "gux_Latn",
    "dga": "dga_Latn",
    "fra": "fra_Latn",
    "eng": "eng_Latn",
    "ff":  "fuv_Latn",
}

# ── Hyperparamètres ───────────────────────────────────────────────────

TRAIN_EPOCHS          = 5
BATCH_SIZE            = 8        # réduis à 4 si Out Of Memory
EVAL_BATCH_SIZE       = 8
LEARNING_RATE         = 5e-5
WARMUP_STEPS          = 500
WEIGHT_DECAY          = 0.01
LR_SCHEDULER          = "cosine"
EARLY_STOP_PATIENCE   = 2
SAVE_TOTAL_LIMIT      = 2

# ── Tokenisation ──────────────────────────────────────────────────────

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256

# ── Nettoyage ─────────────────────────────────────────────────────────

MIN_CHARS            = 5
MAX_CHARS            = 1000
DEDUP                = True
INCLUDE_RADIO        = True
EXCLUDE_NEEDS_REVIEW = True

# Sources exclues de l'entraînement
VALIDATION_SOURCES = {"flores_dev"}
EXCLUDED_SOURCES   = {"bloom_mono"}

# ── Évaluation ────────────────────────────────────────────────────────

EVAL_DATASET   = "facebook/flores"
EVAL_SPLIT     = "dev"
EVAL_LANG_PAIR = ("moore", "francais")

# ── Whisper ───────────────────────────────────────────────────────────

WHISPER_MODEL_SIZE = "large-v3"
WHISPER_BOOTSTRAP  = True

# ── Dossiers à créer ─────────────────────────────────────────────────

REQUIRED_DIRS = [
    "data/raw/audio/moore",
    "data/raw/audio/dioula",
    "data/raw/audio/fulfulde",
    "data/raw/opus",
    "data/raw/transcriptions",
    "data/processed",
    "models",
    "logs/tensorboard",
    "logs/eval",
]