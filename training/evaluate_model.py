# training/evaluate_model.py — VERSION FINALE
"""
Évalue le modèle NLLB-200 (base et fine-tuné) sur FLORES-200 devtest.
Permet de mesurer le gain du fine-tuning de façon objective.

Usage :
    # Évalue le modèle de base Meta
    python training/evaluate_model.py --model facebook/nllb-200-distilled-600M

    # Évalue le modèle fine-tuné
    python training/evaluate_model.py --model models/nllb-burkina-v1

    # Compare les deux
    python training/evaluate_model.py --compare

Métriques calculées :
    - BLEU      : précision n-grammes (standard de traduction)
    - chrF      : F-score sur les caractères (meilleur pour langues rares)
    - TER       : taux d'erreur de traduction (plus bas = meilleur)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate

from training.config import (
    BASE_MODEL,
    OUTPUT_DIR,
    BURKINA_LANGS,
    LANG_CODES,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
)

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("logs/eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Métriques ─────────────────────────────────────────────────────────

bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
ter_metric  = evaluate.load("ter")


# ── Chargement modèle ─────────────────────────────────────────────────

def load_model(model_path: str):
    logger.info(f"Chargement modèle : {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(DEVICE)
    model.eval()
    logger.info(f"✅ Modèle chargé sur {DEVICE}")
    return tokenizer, model


# ── Chargement FLORES-200 ─────────────────────────────────────────────

# def load_flores(src_lang: str, tgt_lang: str) -> list[dict]:
#     """Charge le split devtest de FLORES-200 pour une paire de langues."""
#     try:
#         from datasets import load_dataset
#     except ImportError:
#         raise ImportError("pip install datasets")
#
#     src_nllb = LANG_CODES.get(src_lang, src_lang)
#     tgt_nllb = LANG_CODES.get(tgt_lang, tgt_lang)
#
#     try:
#         ds = load_dataset(
#             "facebook/flores",
#             # f"{src_nllb}-{tgt_nllb}",
#             split="devtest",
#             trust_remote_code=True,
#         )
#     except Exception:
#         # Essaie l'ordre inverse
#         ds = load_dataset(
#             "facebook/flores",
#             f"{tgt_nllb}-{src_nllb}",
#             split="devtest",
#             trust_remote_code=True,
#         )
#
#     pairs = []
#     for row in ds:
#         trans = row.get("translation", {})
#         src = trans.get(src_nllb, "").strip()
#         tgt = trans.get(tgt_nllb, "").strip()
#         if src and tgt:
#             pairs.append({"src": src, "tgt": tgt})
#
#     logger.info(f"✅ FLORES-200 devtest {src_lang}→{tgt_lang} : {len(pairs)} phrases")
#     return pairs


def load_flores(src_lang: str, tgt_lang: str) -> list[dict]:
    from datasets import load_dataset

    src_nllb = LANG_CODES.get(src_lang, src_lang)
    tgt_nllb = LANG_CODES.get(tgt_lang, tgt_lang)

    # Le dataset facebook/flores n'expose que des configs eng_Latn-XXX.
    # On charge la config "all" qui contient toutes les langues en colonnes,
    # puis on extrait nous-mêmes la paire src/tgt voulue.
    ds = load_dataset("facebook/flores", "all", split="devtest")

    pairs = []
    for row in ds:
        src = row.get(f"sentence_{src_nllb}", "").strip()
        tgt = row.get(f"sentence_{tgt_nllb}", "").strip()
        if src and tgt:
            pairs.append({"src": src, "tgt": tgt})

    logger.info(f"✅ FLORES-200 devtest {src_lang}→{tgt_lang} : {len(pairs)} phrases")
    return pairs

# ── Traduction par lot ────────────────────────────────────────────────

def translate_batch(
    tokenizer,
    model,
    texts: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 16,
) -> list[str]:
    src_nllb = LANG_CODES.get(src_lang, src_lang)
    tgt_nllb = LANG_CODES.get(tgt_lang, tgt_lang)
    tgt_id   = tokenizer.convert_tokens_to_ids(tgt_nllb)

    tokenizer.src_lang = src_nllb
    translations = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  Traduction", leave=False):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_length=MAX_TARGET_LENGTH,
                num_beams=5,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


# ── Calcul des métriques ──────────────────────────────────────────────

def compute_metrics(
    predictions: list[str],
    references: list[str],
) -> dict:
    refs_bleu = [[r] for r in references]

    bleu = bleu_metric.compute(predictions=predictions, references=refs_bleu)
    chrf = chrf_metric.compute(predictions=predictions, references=refs_bleu)
    ter  = ter_metric.compute(predictions=predictions, references=refs_bleu)

    return {
        "bleu":   round(bleu["score"], 2),
        "bleu_1": round(bleu["precisions"][0], 2),
        "bleu_2": round(bleu["precisions"][1], 2),
        "chrf":   round(chrf["score"], 2),
        "ter":    round(ter["score"], 2),
    }


# ── Évaluation d'un modèle ────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    lang_pairs: Optional[list[tuple]] = None,
) -> dict:
    """
    Évalue un modèle sur toutes les paires de langues burkinabè.

    Returns:
        dict avec les scores par paire de langue
    """
    tokenizer, model = load_model(model_path)

    if lang_pairs is None:
        lang_pairs = [
            (lang_name, "francais")
            for lang_name, _, _ in BURKINA_LANGS
        ]

    all_results = {}
    model_name = Path(model_path).name

    logger.info(f"\n{'='*55}")
    logger.info(f"Évaluation : {model_name}")
    logger.info(f"{'='*55}")

    for src_lang, tgt_lang in lang_pairs:
        logger.info(f"\n  {src_lang} → {tgt_lang}")

        try:
            # Chargement FLORES-200
            pairs = load_flores(src_lang, tgt_lang)
            if not pairs:
                logger.warning(f"  ⚠️  Pas de données FLORES pour {src_lang}→{tgt_lang}")
                continue

            sources    = [p["src"] for p in pairs]
            references = [p["tgt"] for p in pairs]

            # Traduction
            t0 = time.time()
            predictions = translate_batch(
                tokenizer, model,
                sources, src_lang, tgt_lang,
            )
            duration = round(time.time() - t0, 1)

            # Métriques
            metrics = compute_metrics(predictions, references)
            metrics["duration_s"]   = duration
            metrics["nb_sentences"] = len(pairs)

            all_results[f"{src_lang}_{tgt_lang}"] = metrics

            logger.info(
                f"  BLEU={metrics['bleu']:.1f} | "
                f"chrF={metrics['chrf']:.1f} | "
                f"TER={metrics['ter']:.1f} | "
                f"{len(pairs)} phrases en {duration}s"
            )

            # Affiche quelques exemples
            logger.info("  Exemples :")
            for i in range(min(3, len(predictions))):
                logger.info(f"    SRC : {sources[i][:80]}")
                logger.info(f"    REF : {references[i][:80]}")
                logger.info(f"    PRD : {predictions[i][:80]}")
                logger.info("")

        except Exception as e:
            logger.warning(f"  ⚠️  {src_lang}→{tgt_lang} ignoré : {e}")

    return all_results


# ── Comparaison avant / après fine-tuning ────────────────────────────

def compare_models():
    """Compare le modèle de base et le modèle fine-tuné."""

    lang_pairs = [(lang, "francais") for lang, _, _ in BURKINA_LANGS]

    # Modèle de base
    logger.info("\n🔵 Évaluation du modèle de BASE (Meta NLLB-200 600M)...")
    results_base = evaluate_model(BASE_MODEL, lang_pairs)

    # Modèle fine-tuné
    if not Path(OUTPUT_DIR).exists():
        logger.error(
            f"❌ Modèle fine-tuné introuvable : {OUTPUT_DIR}\n"
            "Lance d'abord : python training/finetune_nllb.py"
        )
        return

    logger.info(f"\n🟢 Évaluation du modèle FINE-TUNÉ ({OUTPUT_DIR})...")
    results_finetuned = evaluate_model(OUTPUT_DIR, lang_pairs)

    # Tableau comparatif
    logger.info(f"\n{'='*65}")
    logger.info("📊 COMPARAISON BASE vs FINE-TUNÉ")
    logger.info(f"{'='*65}")
    logger.info(f"{'Paire':<25} {'Base BLEU':>10} {'FT BLEU':>10} {'Gain':>10} {'chrF FT':>10}")
    logger.info("-" * 65)

    total_gain = 0
    count = 0

    for key in results_base:
        base = results_base[key]
        ft   = results_finetuned.get(key, {})

        if not ft:
            continue

        gain      = ft["bleu"] - base["bleu"]
        total_gain += gain
        count     += 1

        gain_str = f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}"
        gain_col = gain_str

        logger.info(
            f"{key:<25} {base['bleu']:>10.1f} {ft['bleu']:>10.1f} "
            f"{gain_col:>10} {ft['chrf']:>10.1f}"
        )

    if count:
        avg_gain = total_gain / count
        logger.info("-" * 65)
        logger.info(f"{'Gain moyen BLEU':<25} {'':<10} {'':<10} {avg_gain:>+10.1f}")

    logger.info(f"{'='*65}")

    # Sauvegarde résultats
    results = {
        "base":       results_base,
        "finetuned":  results_finetuned,
    }
    output_file = RESULTS_DIR / "comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n💾 Résultats sauvegardés : {output_file}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Évaluation du modèle NLLB-200 sur FLORES-200"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Chemin ou nom du modèle à évaluer",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare le modèle de base et le modèle fine-tuné",
    )
    args = parser.parse_args()

    if args.compare:
        compare_models()
    elif args.model:
        results = evaluate_model(args.model)
        output_file = RESULTS_DIR / f"eval_{Path(args.model).name}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n💾 Résultats sauvegardés : {output_file}")
    else:
        logger.info("Usage :")
        logger.info("  python training/evaluate_model.py --model models/nllb-burkina-v1")
        logger.info("  python training/evaluate_model.py --compare")