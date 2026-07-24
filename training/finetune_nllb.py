# training/finetune_nllb.py — VERSION FINALE
"""
Fine-tuning de NLLB-200 distilled 600M sur les langues burkinabè.

Usage :
    python training/finetune_nllb.py

Prérequis :
    pip install transformers datasets evaluate sacrebleu torch accelerate

Structure attendue :
    data/processed/corpus_burkina.csv  ← généré par collect/download_all_datasets.py

Résultat :
    models/nllb-burkina-v1/            ← modèle fine-tuné prêt à l'emploi
"""

import os
import logging
import time
from pathlib import Path

import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

BASE_MODEL   = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR   = "models/nllb-burkina-v1"
CORPUS_FILE  = "data/processed/corpus_burkina.csv"
LOG_DIR      = "logs/tensorboard"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Codes NLLB pour les langues burkinabè
LANG_CODES = {
    "mos": "mos_Latn",   # mooré
    "dyu": "dyu_Latn",   # dioula
    "fuv": "fuv_Latn",   # fulfulde
    "gux": "gux_Latn",   # gourmantchéma
    "dga": "dga_Latn",   # dagaare
    "fra": "fra_Latn",   # français
    "eng": "eng_Latn",   # anglais
    # codes longs (depuis download_all_datasets.py)
    "moore":        "mos_Latn",
    "dioula":       "dyu_Latn",
    "fulfulde":     "fuv_Latn",
    "gourmantsema": "gux_Latn",
    "dagaare":      "dga_Latn",
    "francais":     "fra_Latn",
    "anglais":      "eng_Latn",
}

# Hyperparamètres
TRAIN_EPOCHS        = 5
BATCH_SIZE          = 8     # réduis à 4 si Out Of Memory
LEARNING_RATE       = 5e-5
WARMUP_STEPS        = 500
MAX_SOURCE_LENGTH   = 256
MAX_TARGET_LENGTH   = 256
EARLY_STOP_PATIENCE = 2

# ── Chargement tokenizer et modèle ───────────────────────────────────

logger.info(f"Chargement de {BASE_MODEL} sur {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
)
model = model.to(DEVICE)
logger.info(f"✅ Modèle chargé — {sum(p.numel() for p in model.parameters()):,} paramètres")

# ── Chargement du corpus ──────────────────────────────────────────────

def load_corpus() -> pd.DataFrame:
    """Charge et nettoie le corpus depuis le CSV."""
    if not Path(CORPUS_FILE).exists():
        raise FileNotFoundError(
            f"Corpus introuvable : {CORPUS_FILE}\n"
            "Lance d'abord : python collect/download_all_datasets.py"
        )

    df = pd.read_csv(CORPUS_FILE, encoding="utf-8")
    logger.info(f"Corpus chargé : {len(df):,} paires brutes")

    # Nettoyage
    df = df.dropna(subset=["src", "tgt"])
    df = df[df["src"].str.strip().str.len() > 5]
    df = df[df["tgt"].str.strip().str.len() > 5]
    df = df[df["src"].str.len() <= 1000]
    df = df[df["tgt"].str.len() <= 1000]

    # Supprime les doublons
    df = df.drop_duplicates(subset=["src", "tgt"])

    # Exclut FLORES+ dev (réservé à la validation)
    df_train = df[df["source"] != "flores_plus_dev"].copy()
    df_valid  = df[df["source"] == "flores_plus_dev"].copy()

    logger.info(f"✅ Après nettoyage :")
    logger.info(f"   Entraînement : {len(df_train):,} paires")
    logger.info(f"   Validation   : {len(df_valid):,} paires")
    logger.info(f"\n{df_train.groupby('src_lang').size().to_string()}")

    return df_train, df_valid


# ── Tokenisation ──────────────────────────────────────────────────────

def tokenize_batch(examples):
    """Tokenise un batch de paires src/tgt."""
    src_lang = examples["src_lang"][0]
    tgt_lang = examples["tgt_lang"][0]

    src_code = LANG_CODES.get(src_lang, src_lang)
    tgt_code = LANG_CODES.get(tgt_lang, tgt_lang)

    tokenizer.src_lang = src_code

    model_inputs = tokenizer(
        examples["src"],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["tgt"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length",
        )

    # Remplace le padding des labels par -100 (ignoré par la loss)
    labels_ids = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels_ids
    return model_inputs


def build_datasets(df_train: pd.DataFrame, df_valid: pd.DataFrame) -> DatasetDict:
    """Construit les datasets HuggingFace depuis les DataFrames."""
    # Split train/test sur df_train si pas assez de données de validation
    if len(df_valid) < 500:
        logger.warning("Peu de données de validation — split 90/10 sur df_train")
        split_idx  = int(len(df_train) * 0.9)
        df_valid   = df_train[split_idx:].copy()
        df_train   = df_train[:split_idx].copy()

    train_ds = Dataset.from_pandas(df_train.reset_index(drop=True))
    valid_ds = Dataset.from_pandas(df_valid.reset_index(drop=True))

    logger.info("Tokenisation du corpus...")
    t0 = time.time()

    tokenized = DatasetDict({
        "train":      train_ds.map(tokenize_batch, batched=True,
                                   remove_columns=train_ds.column_names,
                                   desc="Tokenisation train"),
        "validation": valid_ds.map(tokenize_batch, batched=True,
                                   remove_columns=valid_ds.column_names,
                                   desc="Tokenisation validation"),
    })

    logger.info(f"✅ Tokenisation terminée en {round(time.time()-t0)}s")
    return tokenized


# ── Métriques ─────────────────────────────────────────────────────────

bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )

    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )

    # SacreBLEU attend des références sous forme de liste de listes
    decoded_labels = [[label] for label in decoded_labels]

    result = bleu_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
    return {
        "bleu":       round(result["score"], 2),
        "bleu_1":     round(result["precisions"][0], 2),
        "bleu_2":     round(result["precisions"][1], 2),
    }


# ── Entraînement ──────────────────────────────────────────────────────

def train(tokenized_datasets: DatasetDict):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Epochs et batch
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        # Optimiseur
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        # Précision
        fp16=DEVICE == "cuda",          # float16 sur GPU
        bf16=False,

        # Évaluation et sauvegarde
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,

        # Génération
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,

        # Logs
        logging_dir=LOG_DIR,
        logging_steps=100,
        report_to="tensorboard",

        # Divers
        save_total_limit=2,             # garde les 2 meilleurs checkpoints
        push_to_hub=False,
        dataloader_num_workers=2,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if DEVICE == "cuda" else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)
        ],
    )

    logger.info("🚀 Démarrage du fine-tuning...")
    logger.info(f"   Modèle base  : {BASE_MODEL}")
    logger.info(f"   Device       : {DEVICE}")
    logger.info(f"   Train size   : {len(tokenized_datasets['train']):,}")
    logger.info(f"   Valid size   : {len(tokenized_datasets['validation']):,}")
    logger.info(f"   Epochs       : {TRAIN_EPOCHS}")
    logger.info(f"   Batch size   : {BATCH_SIZE}")
    logger.info(f"   Output       : {OUTPUT_DIR}")

    t0 = time.time()
    trainer.train()
    duration = round((time.time() - t0) / 60, 1)
    logger.info(f"✅ Entraînement terminé en {duration} minutes")

    # Sauvegarde finale
    logger.info("💾 Sauvegarde du modèle final...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"✅ Modèle sauvegardé dans {OUTPUT_DIR}")

    return trainer


# ── Évaluation finale sur FLORES-200 devtest ─────────────────────────

def evaluate_on_flores(trainer: Seq2SeqTrainer):
    """
    Évalue le modèle fine-tuné sur FLORES-200 devtest.
    Ce split n'a jamais été vu pendant l'entraînement.
    """
    try:
        from datasets import load_dataset
        logger.info("📊 Évaluation finale sur FLORES-200 devtest...")

        flores = load_dataset("facebook/flores", "mos_Latn", split="devtest")
        results = trainer.evaluate(flores)
        logger.info(f"🏆 Score BLEU final (mooré) : {results.get('eval_bleu', 'N/A')}")

    except Exception as e:
        logger.warning(f"Évaluation FLORES ignorée : {e}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Chargement corpus
    df_train, df_valid = load_corpus()

    # 2. Tokenisation
    tokenized = build_datasets(df_train, df_valid)

    # 3. Entraînement
    trainer = train(tokenized)

    # 4. Évaluation finale
    evaluate_on_flores(trainer)

    logger.info(
        f"\n{'='*50}\n"
        f"Fine-tuning terminé !\n"
        f"Modèle disponible dans : {OUTPUT_DIR}\n"
        f"\nPour l'utiliser dans FasoConnect :\n"
        f"  Modifie MODEL_ID dans translation/nllb_engine.py :\n"
        f"  MODEL_ID = './{OUTPUT_DIR}'\n"
        f"  Puis : docker compose up --build -d\n"
        f"{'='*50}"
    )