# collect/download_all_datasets.py
from datasets import load_dataset
import pandas as pd

LANG_PAIRS = [
    ("mos_Latn", "fra_Latn"),  # mooré-français
    ("dyu_Latn", "fra_Latn"),  # dioula-français
    ("fuv_Latn", "fra_Latn"),  # fulfulde-français
    ("mos_Latn", "eng_Latn"),  # mooré-anglais
]

all_pairs = []

for src, tgt in LANG_PAIRS:
    print(f"\n📥 Téléchargement {src} ↔ {tgt}...")

    # 1. NLLB-SEED
    try:
        ds = load_dataset("allenai/nllb", f"{src}-{tgt}", split="train")
        for row in ds:
            all_pairs.append({
                "src": row["translation"][src],
                "tgt": row["translation"][tgt],
                "src_lang": src, "tgt_lang": tgt,
                "source": "nllb_mined",
            })
        print(f"  ✅ NLLB mined : {len(ds)} paires")
    except Exception as e:
        print(f"  ⚠️  NLLB mined : {e}")

    # 2. FLORES+ (dev uniquement pour validation)
    try:
        flores = load_dataset("openlanguagedata/flores_plus",
                              f"{src}-{tgt}", split="dev")
        for row in flores:
            all_pairs.append({
                "src": row["sentence_" + src],
                "tgt": row["sentence_" + tgt],
                "src_lang": src, "tgt_lang": tgt,
                "source": "flores_plus_dev",
            })
        print(f"  ✅ FLORES+ dev : {len(flores)} paires")
    except Exception as e:
        print(f"  ⚠️  FLORES+ : {e}")

# Sauvegarde
df = pd.DataFrame(all_pairs)
df.to_csv("data/processed/corpus_burkina.csv", index=False)
print(f"\n📊 Total : {len(df)} paires")
print(df.groupby(["src_lang", "source"]).size())