# training/finetune_nllb.py
import os
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import pandas as pd
import evaluate

# ── Config ────────────────────────────────────────────────
BASE_MODEL  = "facebook/nllb-200-distilled-600M"  # base à fine-tuner
OUTPUT_DIR  = "models/nllb-burkina-v1"
CORPUS_FILE = "data/processed/corpus_burkina.csv"

LANG_CODES = {
    "mos": "mos_Latn",
    "dyu": "dyu_Latn",
    "fuv": "fuv_Latn",
    "fra": "fra_Latn",
}

# ── Chargement tokenizer et modèle ─────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

# ── Préparation du dataset ─────────────────────────────
def preprocess(examples):
    src_lang = LANG_CODES.get(examples["src_lang"][0], "fra_Latn")
    tgt_lang = LANG_CODES.get(examples["tgt_lang"][0], "fra_Latn")

    tokenizer.src_lang = src_lang
    model_inputs = tokenizer(
        examples["src"],
        max_length=256,
        truncation=True,
        padding="max_length",
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["tgt"],
            max_length=256,
            truncation=True,
            padding="max_length",
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Chargement et split du corpus
df = pd.read_csv(CORPUS_FILE).dropna(subset=["src", "tgt"])
df = df[df["src"].str.len() > 5]  # filtre phrases trop courtes

# 80% train / 10% validation / 10% test
train_size = int(len(df) * 0.8)
val_size   = int(len(df) * 0.1)

dataset = DatasetDict({
    "train": Dataset.from_pandas(df[:train_size].reset_index(drop=True)),
    "validation": Dataset.from_pandas(df[train_size:train_size + val_size].reset_index(drop=True)),
    "test": Dataset.from_pandas(df[train_size + val_size:].reset_index(drop=True)),
})

tokenized = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# ── Métriques ──────────────────────────────────────────
bleu = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels[labels == -100] = tokenizer.pad_token_id
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [[l] for l in decoded_labels]
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": round(result["score"], 2)}

# ── Arguments d'entraînement ───────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=8,    # réduis à 4 si OOM
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
    fp16=True,                        # float16 pour RTX 3060
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    logging_dir="logs/tensorboard",
    logging_steps=100,
    report_to="tensorboard",
    push_to_hub=False,
)

# ── Trainer ────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# ── Lancement ──────────────────────────────────────────
print("🚀 Démarrage du fine-tuning...")
trainer.train()

print("💾 Sauvegarde du modèle...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Modèle sauvegardé dans {OUTPUT_DIR}")

# ── Évaluation finale ──────────────────────────────────
results = trainer.evaluate(tokenized["test"])
print(f"\n📊 Score BLEU final sur le test : {results['eval_bleu']}")