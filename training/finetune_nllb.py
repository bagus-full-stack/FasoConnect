# training/finetune_nllb.py — VERSION FINALE
"""
Fine-tuning de NLLB-200 distilled 600M sur les langues burkinabè.

Corrections finales :
  - Charge corpus_burkina_clean.csv (nettoyé)
  - Imports autonomes (sans dépendance à training.config)
  - Filtre automatique des données monolingues (bloom_mono)
  - dataloader_num_workers=0 (Windows compatible)
  - Login HF automatique au démarrage

Usage :
    python training/finetune_nllb.py

Prérequis :
    data/processed/corpus_burkina_clean.csv  <- généré par clean_corpus.py
"""

import os
import sys
import logging
import time
from pathlib import Path
import unicodedata

import torch
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration (autonome) ──────────────────────────────────────────

BASE_MODEL  = "facebook/nllb-200-distilled-600M"
OUTPUT_DIR  = "models/nllb-burkina-v1"
LOG_DIR     = "logs/tensorboard"
CORPUS_FILE = "data/processed/corpus_burkina_clean.csv"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

LANG_CODES = {
    "moore":        "mos_Latn",
    "dioula":       "dyu_Latn",
    "fulfulde":     "fuv_Latn",
    "gourmantsema": "gux_Latn",
    "dagaare":      "dga_Latn",
    "francais":     "fra_Latn",
    "anglais":      "eng_Latn",
    "mos": "mos_Latn",
    "dyu": "dyu_Latn",
    "fuv": "fuv_Latn",
    "fra": "fra_Latn",
    "eng": "eng_Latn",
    "ff":  "fuv_Latn",
}

TRAIN_EPOCHS        = 5
BATCH_SIZE          = 4     # 8
LEARNING_RATE       = 2e-5  # Au lieu de 5e-5
WARMUP_STEPS        = 500
MAX_SOURCE_LENGTH   = 256
MAX_TARGET_LENGTH   = 256
EARLY_STOP_PATIENCE = 5
SAVE_TOTAL_LIMIT    = 2

os.environ["TENSORBOARD_LOGGING_DIR"] = LOG_DIR
# ── Auth HuggingFace ──────────────────────────────────────────────────

def hf_login():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("HuggingFace authentifié")
        except Exception as e:
            logger.warning(f"Login HF : {e}")
    else:
        logger.warning("HF_TOKEN manquant dans .env")

# ── Chargement modèle ─────────────────────────────────────────────────

def load_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import get_peft_model, LoraConfig, TaskType

    logger.info(f"Chargement {BASE_MODEL} sur {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # 1. Charger en float16 pour économiser la RAM/VRAM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(DEVICE)

    # 2. Configuration LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Cible l'attention
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # 3. Application de LoRA au modèle
    model = get_peft_model(model, lora_config)

    # Affichera : "trainable params: ~4,000,000 || all params: 619,000,000 || trainable%: 0.6%"
    model.print_trainable_parameters()

    logger.info(f"Modèle préparé avec LoRA sur {DEVICE}")
    return tokenizer, model

# ── Chargement corpus ─────────────────────────────────────────────────

def remove_accents_and_special_chars(text: str) -> str:
    if not isinstance(text, str): return ""
    text_no_accents = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    replacements = {
        'ɛ': 'e', 'Ɛ': 'E', 'ɔ': 'o', 'Ɔ': 'O', 'ɓ': 'b', 'Ɓ': 'B',
        'ɗ': 'd', 'Ɗ': 'D', 'ŋ': 'n', 'Ŋ': 'N', 'ɲ': 'ny', 'Ɲ': 'Ny',
        'ʋ': 'v', 'Ʋ': 'V', 'ƴ': 'y', 'Ƴ': 'Y'
    }
    for special, standard in replacements.items():
        text_no_accents = text_no_accents.replace(special, standard)
    return text_no_accents

def load_corpus():
    if not Path(CORPUS_FILE).exists():
        logger.error(
            f"Corpus nettoyé introuvable : {CORPUS_FILE}\n"
            "Lance d'abord : python collect/clean_corpus.py"
        )
        sys.exit(1)

    df = pd.read_csv(CORPUS_FILE, encoding="utf-8")
    logger.info(f"Corpus brut chargé : {len(df):,} lignes")

    # Nettoyage défensif
    df = df.dropna(subset=["src", "tgt"])
    df["src"] = df["src"].astype(str).str.strip()
    df["tgt"] = df["tgt"].astype(str).str.strip()
    df = df[df["src"].str.len() > 5]
    df = df[df["tgt"].str.len() > 5]

    # Exclut les données monolingues (bloom_mono)
    if "source" in df.columns:
        df = df[df["source"] != "bloom_mono"]

    if len(df) == 0:
        logger.error(
            "Corpus vide après filtrage.\n"
            "Vérifie que download_all_datasets.py a bien téléchargé des données."
        )
        sys.exit(1)

    # Split : flores_dev → validation, reste → entraînement
    if "source" in df.columns:
        df_valid = df[df["source"] == "flores_dev"].copy()
        df_train = df[df["source"] != "flores_dev"].copy()
    else:
        df_shuffled = df.sample(frac=1, random_state=42)
        split_idx   = int(len(df_shuffled) * 0.9)
        df_train    = df_shuffled[:split_idx]
        df_valid    = df_shuffled[split_idx:]

    # Si pas assez de validation → split 90/10
    if len(df_valid) < 100 and len(df_train) > 200:
        logger.warning("Peu de données de validation — split 90/10")
        df_shuffled = df_train.sample(frac=1, random_state=42)
        split_idx   = int(len(df_shuffled) * 0.9)
        df_valid    = pd.concat(
            [df_valid, df_shuffled[split_idx:]], ignore_index=True
        )
        df_train = df_shuffled[:split_idx]

    logger.info(f"Entraînement : {len(df_train):,} | Validation : {len(df_valid):,}")

    # 1. SWAPPING (Bidirectionnel)
    logger.info("Inversion des paires (Apprentissage bidirectionnel)...")
    df_inverse = df_train.copy()
    df_inverse = df_inverse.rename(columns={"src": "tgt", "tgt": "src", "src_lang": "tgt_lang", "tgt_lang": "src_lang"})
    df_train = pd.concat([df_train, df_inverse], ignore_index=True)

    # 2. BRUIT (Robustesse - 15% des données)
    logger.info("Injection de bruit (tolérance aux fautes de frappe)...")
    df_noise = df_train.sample(frac=0.15, random_state=42).copy()
    df_noise["src"] = df_noise["src"].apply(remove_accents_and_special_chars)
    original_src = df_train.loc[df_noise.index, "src"]
    df_noise = df_noise[df_noise["src"] != original_src]
    df_train = pd.concat([df_train, df_noise], ignore_index=True)

    # 3. OVERSAMPLING (Équilibrage)
    logger.info("Oversampling des langues minoritaires...")
    MIN_SAMPLES = 15000
    balanced_dfs = []
    for lang, group in df_train.groupby("tgt_lang"):
        count = len(group)
        if count < MIN_SAMPLES:
            balanced_group = group.sample(n=MIN_SAMPLES, replace=True, random_state=42)
            balanced_dfs.append(balanced_group)
        else:
            balanced_dfs.append(group)

    df_train = pd.concat(balanced_dfs, ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    if len(df_train) < 500:
        logger.warning(
            f"Corpus d'entraînement très petit ({len(df_train)} paires).\n"
            "   Le fine-tuning risque de ne pas converger.\n"
            "   Enrichis le corpus puis relance."
        )

    return df_train, df_valid

# ── Tokenisation ──────────────────────────────────────────────────────

# def make_tokenize_fn(tokenizer):
#     def tokenize(examples):
#         src_lang = examples["src_lang"][0] if "src_lang" in examples else "anglais"
#         src_code = LANG_CODES.get(src_lang, "eng_Latn")
#         tokenizer.src_lang = src_code
#
#         inputs = tokenizer(
#             examples["src"],
#             max_length=MAX_SOURCE_LENGTH,
#             truncation=True,
#             # padding="max_length",
#         )
#
#         labels = tokenizer(
#             text_target=examples["tgt"],
#             max_length=MAX_TARGET_LENGTH,
#             truncation=True,
#             # padding="max_length",
#         )
#
#
#         # Remplace padding des labels par -100 (ignoré dans la loss)
#         labels_ids = [
#             [(l if l != tokenizer.pad_token_id else -100) for l in label]
#             for label in labels["input_ids"]
#         ]
#         inputs["labels"] = labels_ids
#         return inputs
#
#     return tokenize

def make_tokenize_fn(tokenizer):
    def tokenize(examples):
        input_ids = []
        attention_masks = []
        labels = []

        # On s'assure d'avoir les listes de langues (fallback si la colonne n'existe pas)
        src_langs = examples.get("src_lang", ["moore"] * len(examples["src"]))
        tgt_langs = examples.get("tgt_lang", ["francais"] * len(examples["tgt"]))

        for i in range(len(examples["src"])):
            # 1. On configure le tokenizer avec les VRAIES langues de CETTE phrase
            tokenizer.src_lang = LANG_CODES.get(src_langs[i], "mos_Latn")
            tokenizer.tgt_lang = LANG_CODES.get(tgt_langs[i], "fra_Latn")

            # 2. Tokenisation de la phrase source
            inp = tokenizer(
                examples["src"][i],
                max_length=MAX_SOURCE_LENGTH,
                truncation=True
            )

            # 3. Tokenisation de la traduction (qui inclura désormais le bon token de langue cible)
            lab = tokenizer(
                text_target=examples["tgt"][i],
                max_length=MAX_TARGET_LENGTH,
                truncation=True
            )

            input_ids.append(inp["input_ids"])
            attention_masks.append(inp["attention_mask"])
            labels.append(lab["input_ids"])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }

    return tokenize

def build_datasets(tokenizer, df_train, df_valid):
    from datasets import Dataset, DatasetDict

    logger.info("Tokenisation du corpus...")
    t0 = time.time()

    fn = make_tokenize_fn(tokenizer)

    train_ds = Dataset.from_pandas(df_train.reset_index(drop=True))
    valid_ds = Dataset.from_pandas(df_valid.reset_index(drop=True))

    tokenized = DatasetDict({
        "train": train_ds.map(
            fn, batched=True,
            remove_columns=train_ds.column_names,
            desc="Train",
        ),
        "validation": valid_ds.map(
            fn, batched=True,
            remove_columns=valid_ds.column_names,
            desc="Validation",
        ),
    })

    logger.info(f"Tokenisation terminée en {round(time.time()-t0)}s")
    return tokenized

# ── Métriques ─────────────────────────────────────────────────────────

# def make_compute_metrics(tokenizer):
#     import evaluate
#     bleu = evaluate.load("sacrebleu")
#
#     def compute_metrics(eval_pred):
#         predictions, labels = eval_pred
#         decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#         labels[labels == -100] = tokenizer.pad_token_id
#         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#         decoded_labels = [[l] for l in decoded_labels]
#         result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
#         return {
#             "bleu":   round(result["score"], 2),
#             "bleu_1": round(result["precisions"][0], 2),
#             "bleu_2": round(result["precisions"][1], 2),
#         }
#
#     return compute_metrics


def make_compute_metrics(tokenizer):
    import evaluate
    import numpy as np
    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Parfois, les prédictions sont retournées sous forme de tuple, on prend le premier élément
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # 1. Remplacer les -100 par le pad_token_id dans les PRÉDICTIONS
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # 2. Remplacer les -100 par le pad_token_id dans les LABELS
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Format attendu par sacrebleu : liste de listes pour les références
        decoded_labels = [[l] for l in decoded_labels]

        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "bleu": round(result["score"], 2),
            "bleu_1": round(result["precisions"][0], 2),
            "bleu_2": round(result["precisions"][1], 2),
        }

    return compute_metrics

# ── Entraînement ──────────────────────────────────────────────────────

def train(tokenizer, model, tokenized_datasets):
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Epochs et batch
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Optimiseur
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.05,  # Au lieu de 0.01
        lr_scheduler_type="cosine",

        # Précision
        fp16=DEVICE == "cuda",
        bf16=False,

        # Évaluation et sauvegarde
        eval_strategy="steps",
        eval_steps=1000,    # Évalue le modèle tous les 1000 batchs
        save_strategy="steps",
        save_steps=1000,    # Sauvegarde un checkpoint en même temps
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,

        # Génération
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,

        # Logs
        # logging_dir=LOG_DIR,
        logging_steps=50,
        report_to="tensorboard",

        # Divers
        save_total_limit=SAVE_TOTAL_LIMIT,
        push_to_hub=False,
        # ✅ 0 sur Windows pour éviter les erreurs multiprocessing
        dataloader_num_workers=0,
    )


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,  # au lieu de tokenizer=tokenizer
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if DEVICE == "cuda" else None,
        ),
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )

    logger.info("Démarrage du fine-tuning...")
    logger.info(f"  Modèle  : {BASE_MODEL}")
    logger.info(f"  Device  : {DEVICE}")
    logger.info(f"  Train   : {len(tokenized_datasets['train']):,}")
    logger.info(f"  Valid   : {len(tokenized_datasets['validation']):,}")
    logger.info(f"  Epochs  : {TRAIN_EPOCHS}")
    logger.info(f"  Batch   : {BATCH_SIZE}")
    logger.info(f"  Output  : {OUTPUT_DIR}")
    logger.info(f"  TBoard  : tensorboard --logdir {LOG_DIR}")

    t0 = time.time()
    trainer.train()
    duration = round((time.time() - t0) / 60, 1)
    logger.info(f"Fine-tuning terminé en {duration} minutes")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Modèle sauvegardé : {OUTPUT_DIR}")

    return trainer

# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    hf_login()
    torch.cuda.empty_cache()

    tokenizer, model   = load_model()
    df_train, df_valid = load_corpus()
    tokenized          = build_datasets(tokenizer, df_train, df_valid)
    trainer            = train(tokenizer, model, tokenized)

    logger.info(
        f"\n{'='*50}\n"
        f"Fine-tuning termine !\n"
        f"Modele : {OUTPUT_DIR}\n\n"
        f"Integrer dans FasoConnect :\n"
        f"  translation/nllb_engine.py :\n"
        f"  MODEL_ID = './{OUTPUT_DIR}'\n"
        f"  Puis : docker compose up --build -d\n"
        f"{'='*50}"
    )