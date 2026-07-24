# collect/transcribe_radio.py — VERSION FINALE
"""
Transcription automatique des émissions radio burkinabè avec Whisper.
Génère des paires bilingues (langue locale → français) pour le fine-tuning.

Usage :
    pip install openai-whisper torch soundfile librosa pandas tqdm
    python collect/transcribe_radio.py

Structure attendue :
    data/raw/audio/
        ├── moore/
        │   ├── journal_moore_01.mp3
        │   └── emission_moore_02.mp3
        ├── dioula/
        │   └── journal_dioula_01.mp3
        └── fulfulde/
            └── sante_fulfulde_01.mp3

Résultat :
    data/raw/transcriptions/   ← JSON par fichier audio
    data/processed/corpus_burkina_radio.csv  ← paires bilingues prêtes
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Dossiers ──────────────────────────────────────────────────────────

AUDIO_DIR         = Path("data/raw/audio")
TRANSCRIPTION_DIR = Path("data/raw/transcriptions")
PROCESSED_DIR     = Path("data/processed")

for d in [AUDIO_DIR, TRANSCRIPTION_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Mapping langue → code Whisper ─────────────────────────────────────

# Whisper utilise les codes ISO 639-1 (2 lettres)
WHISPER_LANG_MAP = {
    "moore":        "mos",   # mooré — support limité dans Whisper
    "dioula":       "dyu",   # dioula — support limité
    "fulfulde":     "ff",    # fulfulde — support limité
    "gourmantsema": "gur",   # gourmantchéma — très limité
    "dagaare":      "dga",   # dagaare — très limité
    "francais":     "fr",    # français — excellent
    "anglais":      "en",    # anglais — excellent
}

# Formats audio supportés
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}


# ── Chargement Whisper ────────────────────────────────────────────────

def load_whisper_model(model_size: str = "large-v3"):
    """
    Charge le modèle Whisper.
    Tailles disponibles : tiny, base, small, medium, large-v3
    large-v3 recommandé pour les langues à faibles ressources.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Installe Whisper : pip install openai-whisper")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Chargement Whisper {model_size} sur {device}...")
    model = whisper.load_model(model_size, device=device)
    logger.info(f"✅ Whisper {model_size} prêt")
    return model


# ── Transcription d'un fichier audio ─────────────────────────────────

def transcribe_file(
    model,
    audio_path: Path,
    lang: str,
    task: str = "transcribe",
) -> Optional[dict]:
    """
    Transcrit un fichier audio en texte.

    Args:
        model      : modèle Whisper
        audio_path : chemin vers le fichier audio
        lang       : langue de l'audio (clé de WHISPER_LANG_MAP)
        task       : 'transcribe' (garde langue originale)
                     'translate'  (traduit en anglais)

    Returns:
        dict avec segments, texte complet et métadonnées
    """
    whisper_lang = WHISPER_LANG_MAP.get(lang, lang)

    try:
        t0 = time.time()
        result = model.transcribe(
            str(audio_path),
            language=whisper_lang,
            task=task,
            verbose=False,
            condition_on_previous_text=True,
            temperature=0.0,            # déterministe
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
        )
        duration = round(time.time() - t0, 1)

        return {
            "file":         audio_path.name,
            "lang":         lang,
            "whisper_lang": whisper_lang,
            "task":         task,
            "text":         result["text"].strip(),
            "duration_s":   duration,
            "segments": [
                {
                    "id":    s["id"],
                    "start": round(s["start"], 2),
                    "end":   round(s["end"], 2),
                    "text":  s["text"].strip(),
                }
                for s in result["segments"]
                if s["text"].strip() and len(s["text"].strip()) > 5
            ],
        }

    except Exception as e:
        logger.error(f"  ❌ Erreur transcription {audio_path.name} : {e}")
        return None


# ── Traitement d'un dossier de langue ────────────────────────────────

def process_language_folder(
    model,
    lang: str,
    lang_dir: Path,
) -> list[dict]:
    """
    Traite tous les fichiers audio d'un dossier langue.
    Génère un JSON de transcription par fichier.
    """
    audio_files = [
        f for f in lang_dir.iterdir()
        if f.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if not audio_files:
        logger.warning(f"  Aucun fichier audio dans {lang_dir}")
        return []

    logger.info(f"  {len(audio_files)} fichiers trouvés pour '{lang}'")
    all_segments = []

    for audio_file in tqdm(audio_files, desc=f"  {lang}", leave=False):
        output_json = TRANSCRIPTION_DIR / f"{audio_file.stem}_{lang}.json"

        # Skip si déjà transcrit
        if output_json.exists():
            logger.info(f"    ⏭  {audio_file.name} déjà transcrit — skip")
            with open(output_json, encoding="utf-8") as f:
                result = json.load(f)
        else:
            logger.info(f"    🎙  Transcription de {audio_file.name}...")
            result = transcribe_file(model, audio_file, lang)

            if result is None:
                continue

            # Sauvegarde le JSON
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(
                f"    ✅ {audio_file.name} → "
                f"{len(result['segments'])} segments en {result['duration_s']}s"
            )

        # Collecte les segments
        for seg in result.get("segments", []):
            all_segments.append({
                "src":      seg["text"],
                "src_lang": lang,
                "file":     audio_file.name,
                "start":    seg.get("start"),
                "end":      seg.get("end"),
            })

    return all_segments


# ── Création des paires bilingues ─────────────────────────────────────

def create_bilingual_pairs(
    model,
    segments: list[dict],
) -> list[dict]:
    """
    Pour chaque segment transcrit en langue locale,
    génère sa traduction en français via Whisper (task='translate').

    Note : Whisper translate → anglais uniquement.
    Pour le français, on utilisera NLLB de base comme bootstrap
    puis correction manuelle.
    """
    pairs = []

    logger.info(f"  Création de {len(segments):,} paires bilingues...")

    for seg in tqdm(segments, desc="  Paires bilingues", leave=False):
        src_text = seg["src"]
        if not src_text or len(src_text) < 5:
            continue

        # Tente une traduction bootstrap via Whisper (→ anglais)
        # La traduction française sera faite via NLLB de base
        pairs.append({
            "src":          src_text,
            "tgt":          "",              # à remplir via NLLB ou manuellement
            "src_lang":     seg["src_lang"],
            "tgt_lang":     "francais",
            "source":       "radio_rtb",
            "audio_file":   seg.get("file"),
            "needs_review": True,           # flag pour révision manuelle
        })

    return pairs


def bootstrap_translation_with_nllb(pairs: list[dict]) -> list[dict]:
    """
    Traduit automatiquement les segments en français
    avec le modèle NLLB de base (avant fine-tuning).
    Marque les traductions comme 'needs_review=True' pour révision.
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError:
        logger.warning("transformers non disponible — traductions bootstrap ignorées")
        return pairs

    LANG_CODES = {
        "moore":    "mos_Latn",
        "dioula":   "dyu_Latn",
        "fulfulde": "fuv_Latn",
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("  Chargement NLLB-200 pour bootstrap...")
    tokenizer_nllb = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model_nllb = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M",
        low_cpu_mem_usage=True,
    ).to(device)
    model_nllb.eval()

    fra_id = tokenizer_nllb.convert_tokens_to_ids("fra_Latn")

    enriched = []
    for pair in tqdm(pairs, desc="  Bootstrap NLLB", leave=False):
        src_lang = pair.get("src_lang", "")
        src_code = LANG_CODES.get(src_lang)

        if not src_code or not pair.get("src"):
            enriched.append(pair)
            continue

        try:
            import torch
            tokenizer_nllb.src_lang = src_code
            inputs = tokenizer_nllb(
                pair["src"],
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(device)

            with torch.no_grad():
                output = model_nllb.generate(
                    **inputs,
                    forced_bos_token_id=fra_id,
                    max_length=256,
                    num_beams=4,
                )

            translated = tokenizer_nllb.decode(output[0], skip_special_tokens=True)
            pair["tgt"] = translated
        except Exception:
            pass

        enriched.append(pair)

    logger.info(f"  ✅ Bootstrap NLLB : {len(enriched):,} paires enrichies")
    return enriched


# ── Pipeline principal ────────────────────────────────────────────────

def run_pipeline(
    whisper_model_size: str = "large-v3",
    bootstrap: bool = True,
):
    """
    Pipeline complet :
    1. Transcrit tous les audios par langue
    2. Crée les paires bilingues (src = langue locale)
    3. Bootstrap traduction française via NLLB
    4. Sauvegarde en CSV
    """
    # Vérifie la présence de fichiers audio
    if not AUDIO_DIR.exists() or not any(AUDIO_DIR.iterdir()):
        logger.error(
            f"❌ Dossier audio vide : {AUDIO_DIR}\n"
            "Structure attendue :\n"
            "  data/raw/audio/moore/journal_moore_01.mp3\n"
            "  data/raw/audio/dioula/journal_dioula_01.mp3\n"
            "  etc."
        )
        return

    # Chargement Whisper
    model = load_whisper_model(whisper_model_size)

    all_pairs = []

    # Traite chaque dossier de langue
    for lang_dir in sorted(AUDIO_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue

        lang = lang_dir.name
        if lang not in WHISPER_LANG_MAP:
            logger.warning(f"Langue inconnue : '{lang}' — dossier ignoré")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Traitement : {lang}")
        logger.info(f"{'='*50}")

        segments = process_language_folder(model, lang, lang_dir)
        pairs    = create_bilingual_pairs(model, segments)
        all_pairs.extend(pairs)

        logger.info(f"✅ {lang} : {len(pairs):,} paires extraites")

    if not all_pairs:
        logger.warning("Aucune paire extraite — vérifie tes fichiers audio")
        return

    # Bootstrap traduction
    if bootstrap:
        logger.info(f"\n{'='*50}")
        logger.info("Bootstrap traduction française via NLLB")
        logger.info(f"{'='*50}")
        all_pairs = bootstrap_translation_with_nllb(all_pairs)

    # Sauvegarde
    df = pd.DataFrame(all_pairs)
    df = df[df["src"].str.strip().str.len() > 5]

    output_file = PROCESSED_DIR / "corpus_burkina_radio.csv"
    df.to_csv(output_file, index=False, encoding="utf-8")

    # Stats
    logger.info(f"\n{'='*50}")
    logger.info("📊 RAPPORT FINAL")
    logger.info(f"{'='*50}")
    logger.info(f"Total paires       : {len(df):,}")
    logger.info(f"Avec traduction    : {len(df[df['tgt'].str.len() > 0]):,}")
    logger.info(f"À réviser          : {len(df[df['needs_review'] == True]):,}")
    logger.info(f"Fichier sauvegardé : {output_file}")
    logger.info("\nRépartition par langue :")
    logger.info(df.groupby("src_lang").size().to_string())
    logger.info(f"{'='*50}")
    logger.info(
        "\n⚠️  Les traductions bootstrap sont approximatives.\n"
        "   Une révision manuelle par des locuteurs natifs est fortement recommandée\n"
        "   avant de les intégrer au corpus d'entraînement.\n"
        "   Filtre les paires avec needs_review=True pour les prioriser."
    )


if __name__ == "__main__":
    run_pipeline(
        whisper_model_size="large-v3",
        bootstrap=True,
    )