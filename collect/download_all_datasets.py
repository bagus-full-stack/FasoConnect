# collect/download_all_datasets.py — VERSION 6
"""
Sources confirmées disponibles pour mos/dyu/fuv :
  ✅ Google SMOL (smolsent + smoldoc) — champs src/trg
  ✅ sawadogosalif/MooreFRCollections — mooré-français spécifique
  ✅ sil-ai/bloom-lm — Bloom Library 363 langues
  ✅ openlanguagedata/oldi_seed — NLLB-Seed successeur (fuv_Latn)
  ✅ facebook/flores — validation uniquement
  ✅ allenai/MADLAD-400 — corpus web massif (monolingual bootstrap)
"""

import os
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Auth HF ───────────────────────────────────────────────────────────

def hf_login():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("✅ HuggingFace authentifié")
        except Exception as e:
            logger.warning(f"⚠️  Login HF : {e}")

hf_login()

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
for d in [RAW_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BURKINA_LANGS = [
    ("moore",    "mos", "mos_Latn"),
    ("dioula",   "dyu", "dyu_Latn"),
    ("fulfulde", "fuv", "fuv_Latn"),
]


# ── Source 1 : Google SMOL sentences ─────────────────────────────────

def download_smol_sent() -> list[dict]:
    """Confirmé ✅ — champs src/trg"""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    pairs = []
    configs = {
        "moore":    "smolsent__en_mos",
        "dioula":   "smolsent__en_dyu",
        "fulfulde": "smolsent__en_ff",
    }
    for lang_name, config in configs.items():
        try:
            logger.info(f"  SMOL sent : {config}...")
            ds = load_dataset("google/smol", config, split="train")
            count = 0
            for row in tqdm(ds, desc=f"    {lang_name}", leave=False):
                src = row.get("src", "").strip()
                tgt = row.get("trg", "").strip()
                if src and tgt and len(src) > 5 and len(tgt) > 5:
                    pairs.append({"src": src, "tgt": tgt,
                                  "src_lang": "anglais", "tgt_lang": lang_name,
                                  "source": "smol_sent"})
                    count += 1
            logger.info(f"  ✅ smolsent {lang_name} : {count:,} paires")
        except Exception as e:
            logger.warning(f"  ⚠️  smolsent {config} : {e}")
    return pairs


# ── Source 2 : Google SMOL documents ─────────────────────────────────

def download_smol_doc() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    pairs = []
    configs = {
        "moore": "smoldoc__en_mos",
        "dioula": "smoldoc__en_dyu",
        "fulfulde": "smoldoc__en_ff",
    }
    for lang_name, config in configs.items():
        try:
            logger.info(f"  SMOL doc : {config}...")
            ds = load_dataset("google/smol", config, split="train")
            count = 0

            for row in tqdm(ds, desc=f"    {lang_name}", leave=False):
                # 🛠️ CORRECTION : Utilisation de 'srcs' et 'trgs' (listes de phrases)
                srcs = row.get("srcs", [])
                trgs = row.get("trgs", [])

                # Sécurité : vérifier que les listes existent et sont alignées
                if not srcs or not trgs or len(srcs) != len(trgs):
                    continue

                # Extraction phrase par phrase pour maximiser les paires d'entraînement
                for s, t in zip(srcs, trgs):
                    s = str(s).strip()
                    t = str(t).strip()

                    if s and t and len(s) > 5 and len(t) > 5:
                        pairs.append({
                            "src": s,
                            "tgt": t,
                            "src_lang": "anglais",
                            "tgt_lang": lang_name,
                            "source": "smol_doc"
                        })
                        count += 1

            logger.info(f"  ✅ smoldoc {lang_name} : {count:,} paires extraites")
        except Exception as e:
            logger.warning(f"  ⚠️  smoldoc {config} : {e}")

    return pairs


# ── Source 3 : sawadogosalif/MooreFRCollections ───────────────────────

def download_moorefr() -> list[dict]:
    """
    Dataset bilingue Mooré-Français créé spécifiquement pour le Burkina.
    Découvert sur HuggingFace : sawadogosalif/MooreFRCollections
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    pairs = []
    try:
        logger.info("  MooreFRCollections...")
        ds = load_dataset("sawadogosalif/MooreFRCollections", split="train")

        count = 0
        # Inspecte la première ligne pour trouver les champs
        first = next(iter(ds))
        logger.info(f"  Champs disponibles : {list(first.keys())}")

        ds2 = load_dataset("sawadogosalif/MooreFRCollections", split="train")
        for row in tqdm(ds2, desc="    moorefr", leave=False):
            # Adapte selon les vrais champs trouvés
            src = (row.get("french") or row.get("fra") or
                   row.get("source") or row.get("fr") or "").strip()
            tgt = (row.get("moore") or row.get("mos") or
                   row.get("target") or row.get("mooré") or "").strip()

            if not src or not tgt:
                # Essaie l'ordre inverse
                src, tgt = tgt, src

            if src and tgt and len(src) > 5 and len(tgt) > 5:
                pairs.append({"src": src, "tgt": tgt,
                              "src_lang": "francais", "tgt_lang": "moore",
                              "source": "moorefr_collections"})
                count += 1

        logger.info(f"  ✅ MooreFRCollections : {count:,} paires")
    except Exception as e:
        logger.warning(f"  ⚠️  MooreFRCollections : {e}")
    return pairs


# ── Source 4 : Bloom Library (sil-ai/bloom-lm) ───────────────────────

# def download_bloom_library() -> list[dict]:
#     try:
#         from datasets import load_dataset, get_dataset_config_names
#     except ImportError:
#         return []
#
#     pairs = []
#
#     # On autorise le script distant pour contourner le blocage de sécurité
#     try:
#         configs = get_dataset_config_names("sil-ai/bloom-lm", trust_remote_code=True)
#         logger.info(f"  Bloom Library : {len(configs)} configs disponibles détectées")
#     except Exception as e:
#         logger.warning(f"  ⚠️  Bloom Library configs : {e}")
#         return []
#
#     # Mapping avec les variantes découvertes dans la liste officielle
#     bloom_targets = {
#         "moore": ["mos"],
#         "dioula": ["dyu"],
#         "fulfulde": ["fuv", "fuh", "fub"],  # On ajoute les variantes nigériennes et adamawa !
#     }
#
#     for lang_name, codes in bloom_targets.items():
#         # Cherche si l'un de nos codes correspond à une config Bloom
#         matching = [c for c in configs if any(code in c.lower() for code in codes)]
#
#         if not matching:
#             logger.info(f"  Bloom Library : {lang_name} introuvable.")
#             continue
#
#         for config in matching:
#             try:
#                 logger.info(f"  Bloom Library : Extraction de la variante {config}...")
#
#                 # Autorisation du script distant lors du téléchargement
#                 ds = load_dataset("sil-ai/bloom-lm", config, split="train", trust_remote_code=True)
#
#                 count = 0
#                 for row in tqdm(ds, desc=f"    bloom {config}", leave=False):
#                     text = row.get("text", "").strip()
#                     if text and len(text) > 10:
#                         pairs.append({
#                             "src": text,
#                             "tgt": "",
#                             "src_lang": lang_name,
#                             "tgt_lang": "",
#                             "source": f"bloom_mono_{config}"
#                         })
#                         count += 1
#
#                 logger.info(f"  ✅ Bloom {config} : {count:,} textes (monolingue)")
#             except Exception as e:
#                 logger.warning(f"  ⚠️  Bloom {config} : {str(e)[:80]}")
#
#     return pairs


# ── Source 5 : openlanguagedata/oldi_seed (successeur NLLB-Seed: CORRIGÉ avec ID matching) ─────

def download_oldi_seed() -> list[dict]:
    """
    OLDI Seed — Les traductions sont liées par le champ 'id' entre différentes configs.
    Couvre fuv_Latn (fulfulde).
    """
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        return []

    pairs = []

    try:
        configs = get_dataset_config_names("openlanguagedata/oldi_seed")
    except Exception as e:
        logger.warning(f"  ⚠️  OLDI Seed configs : {e}")
        return []

    # 1. Étape cruciale : Charger l'anglais comme dictionnaire de référence
    eng_dict = {}
    try:
        logger.info("  OLDI Seed : Téléchargement de l'anglais (eng_Latn) pour le matching ID...")
        ds_eng = load_dataset("openlanguagedata/oldi_seed", "eng_Latn", split="train")
        for row in ds_eng:
            # On stocke { id: phrase_en_anglais }
            if "id" in row and "text" in row:
                eng_dict[row["id"]] = row["text"].strip()
        logger.info(f"  ✅ Dictionnaire anglais prêt : {len(eng_dict):,} phrases de référence")
    except Exception as e:
        logger.warning(f"  ⚠️  OLDI Seed (eng_Latn) échoué : {str(e)[:80]}")
        return []  # On arrête si on n'a pas l'anglais pour faire les paires

    # 2. Chercher nos langues cibles
    targets = {
        "moore": ["mos_Latn", "mos-Latn", "mos"],
        "dioula": ["dyu_Latn", "dyu-Latn", "dyu"],
        "fulfulde": ["fuv_Latn", "fuv-Latn", "fuv"],
    }

    for lang_name, codes in targets.items():
        config_found = next((c for c in configs if any(code in c for code in codes)), None)
        if not config_found:
            continue  # Passe au suivant si la langue n'est pas dans le dataset

        try:
            logger.info(f"  OLDI Seed : Match avec {config_found}...")
            ds2 = load_dataset("openlanguagedata/oldi_seed", config_found, split="train")

            count = 0
            for row in ds2:
                row_id = row.get("id")
                tgt = row.get("text", "").strip()

                # 3. La magie opère : on récupère l'anglais grâce à l'ID !
                src = eng_dict.get(row_id, "")

                if src and tgt and len(src) > 5 and len(tgt) > 5:
                    pairs.append({
                        "src": src,
                        "tgt": tgt,
                        "src_lang": "anglais",
                        "tgt_lang": lang_name,
                        "source": "oldi_seed"
                    })
                    count += 1

            logger.info(f"  ✅ OLDI Seed {lang_name} : {count:,} paires trouvées par ID")
        except Exception as e:
            logger.warning(f"  ⚠️  OLDI Seed {config_found} : {str(e)[:80]}")

    return pairs

# ── Source 6 : facebook/flores dev (validation) ───────────────────────

def download_flores_validation() -> list[dict]:
    """Confirmé ✅ — 997 phrases par langue"""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    pairs = []
    flores_langs = {
        "moore":    "mos_Latn",
        "dioula":   "dyu_Latn",
        "fulfulde": "fuv_Latn",
    }

    try:
        ds_fra  = load_dataset("facebook/flores", "fra_Latn", split="dev")
        fra_txt = [r.get("sentence", "").strip() for r in ds_fra]
    except Exception as e:
        logger.warning(f"  ⚠️  FLORES français : {str(e)[:80]}")
        return []

    for lang_name, lang_nllb in flores_langs.items():
        try:
            ds = load_dataset("facebook/flores", lang_nllb, split="dev")
            count = 0
            for i, row in enumerate(ds):
                src = row.get("sentence", "").strip()
                tgt = fra_txt[i] if i < len(fra_txt) else ""
                if src and tgt:
                    pairs.append({"src": src, "tgt": tgt,
                                  "src_lang": lang_name, "tgt_lang": "francais",
                                  "source": "flores_dev"})
                    count += 1
            logger.info(f"  ✅ FLORES dev {lang_name} : {count:,} phrases")
        except Exception as e:
            logger.warning(f"  ⚠️  FLORES {lang_name} : {str(e)[:80]}")

    return pairs


# ── Consolidation ─────────────────────────────────────────────────────

def build_corpus():
    all_pairs = []

    sources = [
        ("1/6 — Google SMOL sentences ✅",        download_smol_sent),
        ("2/6 — Google SMOL documents ✅",         download_smol_doc),
        ("3/6 — MooreFRCollections (mooré-fra)",   download_moorefr),
        ("4/6 — Bloom Library (sil-ai)",           None), # download_bloom_library
        ("5/6 — OLDI Seed (successeur NLLB-Seed)", download_oldi_seed),
        ("6/6 — facebook/flores dev (validation)", download_flores_validation),
    ]

    for label, fn in sources:
        logger.info(f"\n{'='*55}")
        logger.info(label)
        logger.info("="*55)
        try:
            result = fn()
            all_pairs.extend(result)
            logger.info(f"  Sous-total : {len(all_pairs):,} paires cumulées")
        except Exception as e:
            logger.error(f"  Source ignorée : {e}")

    if not all_pairs:
        logger.error("❌ Aucune paire collectée")
        return

    df = pd.DataFrame(all_pairs)
    df = df.dropna(subset=["src"])
    df["src"] = df["src"].str.strip()
    df["tgt"] = df["tgt"].fillna("").str.strip()

    # Sépare bilingues et monolingues
    df_bilingual = df[df["tgt"].str.len() > 5].copy()
    df_bilingual = df_bilingual.drop_duplicates(subset=["src", "tgt"])

    output_file = PROCESSED_DIR / "corpus_burkina.csv"
    df_bilingual.to_csv(output_file, index=False, encoding="utf-8")

    # Sauvegarde aussi les données monolingues (utiles pour évaluation)
    df_mono = df[df["tgt"].str.len() == 0].copy()
    if len(df_mono) > 0:
        mono_file = PROCESSED_DIR / "corpus_burkina_mono.csv"
        df_mono.to_csv(mono_file, index=False, encoding="utf-8")
        logger.info(f"\n💾 Corpus monolingue : {len(df_mono):,} textes → {mono_file}")

    train_df = df_bilingual[df_bilingual["source"] != "flores_dev"]
    valid_df  = df_bilingual[df_bilingual["source"] == "flores_dev"]

    logger.info(f"\n{'='*55}")
    logger.info("📊 RAPPORT FINAL")
    logger.info("="*55)
    logger.info(f"Bilingue total    : {len(df_bilingual):,} paires")
    logger.info(f"  Entraînement    : {len(train_df):,}")
    logger.info(f"  Validation      : {len(valid_df):,} (FLORES dev)")
    logger.info(f"Fichier           : {output_file}")

    if len(train_df) > 0:
        logger.info("\nRépartition par langue :")
        logger.info(train_df.groupby("src_lang").size().to_string())
        logger.info("\nRépartition par source :")
        logger.info(df_bilingual.groupby("source").size().to_string())
    logger.info("="*55)


if __name__ == "__main__":
    build_corpus()