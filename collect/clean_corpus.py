# collect/clean_corpus.py — VERSION CORRIGEE
"""
Fix : suppression de l'import 'from training.config import ...'
      remplacé par des constantes définies directement dans ce fichier.

Nettoyage, déduplication et filtrage qualité du corpus burkinabè.

Usage :
    python collect/clean_corpus.py

Entrées :
    data/processed/corpus_burkina.csv
    data/processed/corpus_burkina_radio.csv  (optionnel)

Sortie :
    data/processed/corpus_burkina_clean.csv
"""

import re
import sys
import logging
import unicodedata
from pathlib import Path

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constantes (remplace l'import de training.config) ────────────────

CORPUS_FILE       = "data/processed/corpus_burkina.csv"
RADIO_CORPUS_FILE = "data/processed/corpus_burkina_radio.csv"
OUTPUT_FILE       = Path("data/processed/corpus_burkina_clean.csv")

MIN_CHARS            = 5
MAX_CHARS            = 1000
DEDUP                = True
INCLUDE_RADIO        = True
EXCLUDE_NEEDS_REVIEW = True

# ── Étape 1 — Chargement ─────────────────────────────────────────────

def load_all_corpora() -> pd.DataFrame:
    frames = []

    if Path(CORPUS_FILE).exists():
        df = pd.read_csv(CORPUS_FILE, encoding="utf-8")
        frames.append(df)
        logger.info(f"✅ Corpus principal : {len(df):,} paires")
    else:
        logger.error(f"❌ Corpus introuvable : {CORPUS_FILE}")
        logger.error("   Lance d'abord : python collect/download_all_datasets.py")
        sys.exit(1)

    if INCLUDE_RADIO and Path(RADIO_CORPUS_FILE).exists():
        df_radio = pd.read_csv(RADIO_CORPUS_FILE, encoding="utf-8")
        if EXCLUDE_NEEDS_REVIEW and "needs_review" in df_radio.columns:
            before = len(df_radio)
            df_radio = df_radio[df_radio["needs_review"] != True]
            logger.info(f"✅ Corpus radio : {len(df_radio):,} paires ({before-len(df_radio):,} exclues)")
        else:
            logger.info(f"✅ Corpus radio : {len(df_radio):,} paires")
        frames.append(df_radio)
    else:
        logger.info("ℹ️  Corpus radio absent ou désactivé")

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"📊 Total brut : {len(df):,} paires")
    return df


# ── Étape 2 — Nettoyage texte ─────────────────────────────────────────

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = "".join(c for c in text if not unicodedata.category(c).startswith("C"))
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_texts(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Nettoyage des textes...")
    df = df.copy()
    df["src"] = df["src"].apply(normalize_text)
    df["tgt"] = df["tgt"].apply(normalize_text)
    return df


# ── Étape 3 — Filtres longueur ────────────────────────────────────────

def filter_by_length(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["src", "tgt"])
    df = df[df["src"].str.len() >= MIN_CHARS]
    df = df[df["tgt"].str.len() >= MIN_CHARS]
    df = df[df["src"].str.len() <= MAX_CHARS]
    df = df[df["tgt"].str.len() <= MAX_CHARS]

    # Filtre ratio longueur (évite les mauvais alignements)
    df["len_ratio"] = df.apply(
        lambda r: max(len(r["src"]), len(r["tgt"])) /
                  max(min(len(r["src"]), len(r["tgt"])), 1),
        axis=1,
    )
    df = df[df["len_ratio"] <= 5.0].drop(columns=["len_ratio"])

    logger.info(f"✅ Filtre longueur : {before:,} → {len(df):,} ({before-len(df):,} supprimées)")
    return df


# ── Étape 4 — Filtres qualité ─────────────────────────────────────────

def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df[df["src"] != df["tgt"]]
    df = df[df["src"].str.count(r"\d") / df["src"].str.len().clip(lower=1) < 0.3]
    df = df[df["src"].str.contains(r"[^\W\d_]", regex=True, na=False)] # r"[a-zA-ZÀ-ÿ]"
    df = df[df["tgt"].str.contains(r"[^\W\d_]", regex=True, na=False)]
    code_pattern = r"[{}\[\]<>|\\=]|http[s]?://"
    df = df[~df["src"].str.contains(code_pattern, regex=True, na=False)]
    df = df[~df["tgt"].str.contains(code_pattern, regex=True, na=False)]

    logger.info(f"✅ Filtre qualité : {before:,} → {len(df):,} ({before-len(df):,} supprimées)")
    return df


# ── Étape 5 — Déduplication ───────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if not DEDUP:
        return df
    before = len(df)

    df = df.drop_duplicates(subset=["src", "tgt"])

    priority_order = ["flores_dev", "moorefr_collections", "oldi_seed", "google_smol", "smol_sent", "smol_doc",
                      "nllb_mined", "nllb_seed", "bible_uedin", "bible_corpus",
                      "jw300", "opus100", "radio_rtb", "masakhane"]

    df["_priority"] = df["source"].apply(
        lambda s: priority_order.index(s) if s in priority_order else 99
    )
    # FIX : inclure tgt_lang dans la clé — sinon une phrase source EN
    # partagée entre mooré/dioula/fulfulde n'en garde qu'une seule
    df = df.sort_values("_priority").drop_duplicates(subset=["src", "tgt_lang"], keep="first")
    df = df.drop(columns=["_priority"])

    logger.info(f"✅ Déduplication : {before:,} → {len(df):,} ({before-len(df):,} doublons)")
    return df

# ── Étape 6 — Split train / validation ───────────────────────────────

def split_train_validation(df: pd.DataFrame):
    df_valid = df[df["source"] == "flores_dev"].copy()
    df_train = df[df["source"] != "flores_dev"].copy()

    if len(df_valid) < 500 and len(df_train) > 100:
        logger.info("ℹ️  Split 90/10 appliqué (FLORES dev insuffisant)")
        df_shuffled = df_train.sample(frac=1, random_state=42)
        split_idx   = int(len(df_shuffled) * 0.9)
        df_valid    = pd.concat([df_valid, df_shuffled[split_idx:]], ignore_index=True)
        df_train    = df_shuffled[:split_idx]

    return df_train, df_valid


# ── Rapport ───────────────────────────────────────────────────────────

def print_report(df_train: pd.DataFrame, df_valid: pd.DataFrame):
    logger.info(f"\n{'='*55}")
    logger.info("📊 RAPPORT DE NETTOYAGE")
    logger.info(f"{'='*55}")
    logger.info(f"Total final       : {len(df_train) + len(df_valid):,} paires")
    logger.info(f"  Entraînement    : {len(df_train):,}")
    logger.info(f"  Validation      : {len(df_valid):,}")

    if len(df_train) > 0:
        logger.info("\nRépartition par langue :")
        logger.info(df_train.groupby("src_lang").size().sort_values(ascending=False).to_string())
        logger.info("\nRépartition par source :")
        logger.info(df_train.groupby("source").size().sort_values(ascending=False).to_string())

    if len(df_train) < 1000:
        logger.warning(
            "\n⚠️  Corpus d'entraînement insuffisant (<1000 paires).\n"
            "   Le fine-tuning ne convergera pas correctement.\n"
            "   Résous d'abord le problème Google SMOL :\n"
            "     python debug_smol.py"
        )
    logger.info("="*55)


# ── Main ──────────────────────────────────────────────────────────────

def run():
    logger.info("🧹 Démarrage du nettoyage...\n")

    df = load_all_corpora()
    df = clean_texts(df)
    df = filter_by_length(df)
    df = filter_quality(df)
    df = deduplicate(df)

    df_train, df_valid = split_train_validation(df)

    df_final = pd.concat([df_train, df_valid], ignore_index=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logger.info(f"\n💾 Corpus nettoyé sauvegardé : {OUTPUT_FILE}")

    print_report(df_train, df_valid)


if __name__ == "__main__":
    run()