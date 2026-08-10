# collect/download_all_datasets.py — VERSION 12 (Corrections de clés et d'URL)
"""
Script de collecte de données modulaire.
Corrections :
 - ARPRIM : utilise les clés 'f' (Français) et 'p' (Pulaar/Fulfulde).
 - Suppression du dataset DS4H-ICTU (supprimé de HuggingFace).
 - Stabilisation de Djelia et RobotsMali via load_dataset direct.
"""

import os
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
for d in [RAW_DIR, PROCESSED_DIR]: d.mkdir(parents=True, exist_ok=True)


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


# ── 1. Google SMOL Sentences ──────────────────────────────────────────
def download_smol_sent() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    pairs, configs = [], {"moore": "smolsent__en_mos", "dioula": "smolsent__en_dyu", "fulfulde": "smolsent__en_ff"}
    for lang, config in configs.items():
        try:
            ds = load_dataset("google/smol", config, split="train")
            for row in tqdm(ds, desc=f"    {lang}", leave=False):
                src, tgt = str(row.get("src", "")).strip(), str(row.get("trg", "")).strip()
                if src and tgt and src != "None" and len(src) > 5 and len(tgt) > 5:
                    pairs.append(
                        {"src": src, "tgt": tgt, "src_lang": "anglais", "tgt_lang": lang, "source": "smol_sent"})
        except Exception:
            pass
    return pairs


# ── 2. Google SMOL Documents ──────────────────────────────────────────
def download_smol_doc() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    pairs, configs = [], {"moore": "smoldoc__en_mos", "dioula": "smoldoc__en_dyu", "fulfulde": "smoldoc__en_ff"}
    for lang, config in configs.items():
        try:
            ds = load_dataset("google/smol", config, split="train")
            for row in tqdm(ds, desc=f"    {lang}", leave=False):
                srcs, trgs = row.get("srcs", []), row.get("trgs", [])
                if srcs and trgs and len(srcs) == len(trgs):
                    for s, t in zip(srcs, trgs):
                        s, t = str(s).strip(), str(t).strip()
                        if s and t and s != "None" and len(s) > 5 and len(t) > 5:
                            pairs.append(
                                {"src": s, "tgt": t, "src_lang": "anglais", "tgt_lang": lang, "source": "smol_doc"})
        except Exception:
            pass
    return pairs


# ── 3. MooreFRCollections (Mooré) ─────────────────────────────────────
def download_moorefr() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    pairs = []
    try:
        logger.info("  Téléchargement sawadogosalif/MooreFRCollections...")
        ds = load_dataset("sawadogosalif/MooreFRCollections", split="train")
        for row in tqdm(ds, desc="    moorefr", leave=False):
            src, tgt = str(row.get("french", "")).strip(), str(row.get("moore", "")).strip()
            if src and tgt and src != "None" and len(src) > 5 and len(tgt) > 5:
                pairs.append({"src": src, "tgt": tgt, "src_lang": "francais", "tgt_lang": "moore", "source": "moorefr"})
    except Exception:
        pass
    return pairs


# ── 4. OLDI Seed (NLLB-Seed) ──────────────────────────────────────────
def download_oldi_seed() -> list[dict]:
    try:
        from datasets import load_dataset, get_dataset_config_names
    except ImportError:
        return []
    pairs, eng_dict = [], {}
    try:
        configs = get_dataset_config_names("openlanguagedata/oldi_seed")
    except Exception:
        return []
    try:
        ds_eng = load_dataset("openlanguagedata/oldi_seed", "eng_Latn", split="train")
        for row in ds_eng: eng_dict[row.get("id")] = row.get("text", "").strip()
    except Exception:
        return []
    targets = {"moore": ["mos"], "dioula": ["dyu"], "fulfulde": ["fuv"]}
    for lang, codes in targets.items():
        conf = next((c for c in configs if any(code in c for code in codes)), None)
        if not conf: continue
        try:
            ds2 = load_dataset("openlanguagedata/oldi_seed", conf, split="train")
            for row in ds2:
                tgt, src = row.get("text", "").strip(), eng_dict.get(row.get("id"), "")
                if src and tgt and src != "None" and len(src) > 5 and len(tgt) > 5:
                    pairs.append(
                        {"src": src, "tgt": tgt, "src_lang": "anglais", "tgt_lang": lang, "source": "oldi_seed"})
        except Exception:
            pass
    return pairs


# ── 5. UVCI Koumankan (Dioula) ────────────────────────────────────────
def download_uvci_dyu() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    pairs = []
    try:
        logger.info("  Téléchargement uvci/Koumankan_mt_dyu_fr...")
        ds = load_dataset("uvci/Koumankan_mt_dyu_fr", split="train")
        for row in tqdm(ds, desc="    uvci", leave=False):
            d = row.get("translation") if "translation" in row else row
            src, tgt = str(d.get("fr", "")).strip(), str(d.get("dyu", "")).strip()
            if src and tgt and src != "None" and len(src) > 5 and len(tgt) > 5:
                pairs.append(
                    {"src": src, "tgt": tgt, "src_lang": "francais", "tgt_lang": "dioula", "source": "hf_uvci"})
    except Exception as e:
        logger.warning(f"  ⚠️  UVCI : {e}")
    return pairs


# ── 6. Djelia Bambara (converti Dioula) ───────────────────────────────
def download_djelia_bm() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    pairs = []
    try:
        logger.info("  Téléchargement djelia/bambara-mt-dataset...")
        # On force la lecture par défaut sans script distant
        ds = load_dataset("djelia/bambara-mt-dataset", split="train")
        for row in tqdm(ds, desc="    djelia", leave=False):
            sl, tl = str(row.get("source_lang", "")), str(row.get("target_lang", ""))
            st, tt = str(row.get("source_text", "")).strip(), str(row.get("target_text", "")).strip()

            # On prend tout ce qui relie Bam <-> Fra, et on taggue le Bam comme "dioula"
            if "fra" in sl and "bam" in tl:
                if st and tt and st != "None" and len(st) > 5 and len(tt) > 5:
                    pairs.append(
                        {"src": st, "tgt": tt, "src_lang": "francais", "tgt_lang": "dioula", "source": "hf_djelia"})
            elif "bam" in sl and "fra" in tl:
                if st and tt and st != "None" and len(st) > 5 and len(tt) > 5:
                    pairs.append(
                        {"src": tt, "tgt": st, "src_lang": "francais", "tgt_lang": "dioula", "source": "hf_djelia"})
    except Exception as e:
        logger.warning(f"  ⚠️  Djelia : {e}")
    return pairs


# ── 7. RobotsMali Bambara (Bambara -> Dyu) ───────────────────────────

def download_robotsmali_bm() -> list[dict]:
    """
    load_dataset("RobotsMaliAI/bayelemabaga", ...) échoue systématiquement
    avec "Dataset scripts are no longer supported" — les versions récentes
    de la lib `datasets` bloquent TOUT dataset qui utilise un script de
    chargement (.py), quel que soit le config/split demandé. Aucun
    paramètre de load_dataset() ne contourne ça.

    Solution : télécharger directement l'archive tar.gz publiée par
    RobotsMali (la même source que celle utilisée par leur propre script
    HF) et parser les fichiers texte alignés nous-mêmes, sans passer par
    la lib `datasets`.

    Structure de l'archive (confirmée via bayelemabaga.py) :
      bayelemabaga/train/train.bam, train.fr
      bayelemabaga/valid/dev.bam,   dev.fr
      bayelemabaga/test/test.bam,   test.fr
    """
    import io
    import tarfile
    import requests

    ARCHIVE_URL = "https://raw.githubusercontent.com/RobotsMali-AI/datasets/master/bayelemabaga.tar.gz"
    pairs = []

    try:
        logger.info("  Téléchargement direct de l'archive bayelemabaga...")
        resp = requests.get(ARCHIVE_URL, timeout=60)
        resp.raise_for_status()

        # Le fichier s'appelle .tar.gz mais n'est pas toujours réellement
        # compressé en gzip (constaté : contenu tar brut sous ce nom).
        # "r:*" laisse tarfile auto-détecter gzip / bzip2 / xz / aucun.
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:*") as tar:
            members = {m.name: m for m in tar.getmembers()}

            def lire(nom_partiel):
                match = next((n for n in members if n.endswith(nom_partiel)), None)
                if not match:
                    return None
                f = tar.extractfile(members[match])
                return f.read().decode("utf-8").strip().split("\n") if f else None

            splits = [
                ("train/train.bam", "train/train.fr"),
                ("valid/dev.bam", "valid/dev.fr"),
                ("test/test.bam", "test/test.fr"),
            ]

            count = 0
            for bam_path, fr_path in splits:
                bam_lines = lire(bam_path)
                fr_lines = lire(fr_path)
                if not bam_lines or not fr_lines:
                    logger.warning(f"  ⚠️  Fichiers introuvables : {bam_path} / {fr_path}")
                    continue
                for bam, fr in zip(bam_lines, fr_lines):
                    bam, fr = bam.strip(), fr.strip()
                    if bam and fr and len(bam) > 1 and len(fr) > 1:
                        pairs.append({"src": fr, "tgt": bam,
                                      "src_lang": "francais", "tgt_lang": "bambara",
                                      "source": "hf_robotsmali"})
                        count += 1

        logger.info(f"  ✅ Extrait : {count:,} paires (bambara — proche du dioula)")
    except requests.RequestException as e:
        logger.warning(f"  ⚠️  Échec téléchargement archive : {e}")
    except Exception as e:
        logger.warning(f"  ⚠️  bayelemabaga (archive) : {e}")

    return pairs


# ── 8. ARPRIM Pulaar (Fulfulde) ───────────────────────────────────────
def download_arprim_fuv() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    pairs = []
    try:
        logger.info("  Téléchargement ARPRIM/pulaar_fulfulde...")
        ds = load_dataset("ARPRIM/pulaar_fulfulde", split="train")
        for row in tqdm(ds, desc="    arprim", leave=False):
            # CORRECTION : Utilisation des clés 'f' (français) et 'p' (pulaar) basées sur vos logs
            src, tgt = str(row.get("f", "")).strip(), str(row.get("p", "")).strip()
            if src and tgt and src != "None" and len(src) > 5 and len(tgt) > 5:
                pairs.append(
                    {"src": src, "tgt": tgt, "src_lang": "francais", "tgt_lang": "fulfulde", "source": "hf_arprim"})
    except Exception as e:
        logger.warning(f"  ⚠️  ARPRIM : {e}")
    return pairs


# ── 9. FLORES Dev (Validation) ────────────────────────────────────────
def download_flores_validation() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    pairs, flores_langs = [], {"moore": "mos_Latn", "dioula": "dyu_Latn", "fulfulde": "fuv_Latn"}
    try:
        ds_fra = load_dataset("facebook/flores", "fra_Latn", split="dev")
        fra_txt = [r.get("sentence", "").strip() for r in ds_fra]
    except Exception:
        return []
    for lang, nllb_code in flores_langs.items():
        try:
            ds = load_dataset("facebook/flores", nllb_code, split="dev")
            for i, row in enumerate(ds):
                src, tgt = row.get("sentence", "").strip(), fra_txt[i] if i < len(fra_txt) else ""
                if src and tgt and src != "None":
                    pairs.append(
                        {"src": src, "tgt": tgt, "src_lang": lang, "tgt_lang": "francais", "source": "flores_dev"})
        except Exception:
            pass
    return pairs


# ── 10. TICO-19 Santé (Fulfulde) ──────────────────────────────────────
# def download_tico19_fuv() -> list[dict]:
#     try:
#         from datasets import load_dataset
#     except ImportError:
#         return []
#
#     pairs = []
#     logger.info("  Téléchargement atepeq/tico19 (Santé / Fulfulde)...")
#
#     try:
#         # Chargement du dataset Parquet natif sans script Python défectueux
#         ds = load_dataset("atepeq/tico19", split="train")
#
#         for row in tqdm(ds, desc="    tico19", leave=False):
#             d = row.get("translation") if "translation" in row else row
#
#             # Extraction des clés Anglais (en) et Fulfulde (fuv)
#             src = str(d.get("en") or d.get("source") or "").strip()
#             tgt = str(d.get("fuv") or d.get("target") or "").strip()
#
#             if src and tgt and src != "None" and len(src) > 5 and len(tgt) > 5:
#                 pairs.append({
#                     "src": src,
#                     "tgt": tgt,
#                     "src_lang": "anglais",
#                     "tgt_lang": "fulfulde",
#                     "source": "hf_tico19"
#                 })
#
#         logger.info(f"  ✅ {len(pairs):,} paires médicales/santé récupérées")
#     except Exception as e:
#         logger.warning(f"  ⚠️  Erreur TICO-19 : {e}")
#
#     return pairs

def download_tico19_fuv() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    pairs = []
    logger.info("  Téléchargement atepeq/tico19 (Santé / Fulfulde)...")

    try:
        # On charge directement le split "en_fuv"
        ds = load_dataset("atepeq/tico19", split="en_fuv")

        for row in tqdm(ds, desc="    tico19", leave=False):
            # D'après la documentation exacte du dataset :
            src = str(row.get("source", "")).strip()
            tgt = str(row.get("translation", "")).strip()

            if src and tgt and src != "None" and len(src) > 5 and len(tgt) > 5:
                pairs.append({
                    "src": src,
                    "tgt": tgt,
                    "src_lang": "anglais",
                    "tgt_lang": "fulfulde",
                    "source": "hf_tico19"
                })

        logger.info(f"  ✅ {len(pairs):,} paires médicales/santé récupérées")
    except Exception as e:
        logger.warning(f"  ⚠️  Erreur TICO-19 : {e}")

    return pairs


# ── CONSOLIDATION GLOBALE ─────────────────────────────────────────────
def build_corpus():
    all_pairs = []

    sources = [
        ("1/9 — Google SMOL Sentences", download_smol_sent),
        ("2/9 — Google SMOL Documents", download_smol_doc),
        ("3/9 — MooreFRCollections", download_moorefr),
        ("4/9 — OLDI Seed (NLLB-Seed)", download_oldi_seed),
        ("5/9 — UVCI Koumankan (Dioula)", download_uvci_dyu),
        ("6/9 — Djelia (Bambara -> Dioula)", download_djelia_bm),
        ("7/9 — RobotsMali (Bambara -> Dyu)", download_robotsmali_bm),
        ("8/9 — ARPRIM (Fulfulde / Pulaar)", download_arprim_fuv),
        ("9/9 — facebook/flores (Validation)", download_flores_validation),
        ("10/10 — TICO-19 Santé (Fulfulde)", download_tico19_fuv),
    ]

    for label, fn in sources:
        logger.info(f"\n{'=' * 55}\n{label}\n{'=' * 55}")
        try:
            result = fn()
            all_pairs.extend(result)
            logger.info(f"  ✅ Extrait : {len(result):,} paires (Sous-total : {len(all_pairs):,})")
        except Exception as e:
            logger.error(f"  ❌ Source ignorée : {e}")

    df = pd.DataFrame(all_pairs)
    if len(df) == 0:
        logger.error("Aucune paire collectée ! Fin du script.")
        return

    df = df.dropna(subset=["src", "tgt"])
    df["src"] = df["src"].str.strip()
    df["tgt"] = df["tgt"].str.strip()

    df_bilingual = df[(df["src"].str.len() > 5) & (df["tgt"].str.len() > 5)].copy()
    df_bilingual = df_bilingual.drop_duplicates(subset=["src", "tgt"])

    output_file = PROCESSED_DIR / "corpus_burkina.csv"
    df_bilingual.to_csv(output_file, index=False, encoding="utf-8")

    train_df = df_bilingual[df_bilingual["source"] != "flores_dev"]
    valid_df = df_bilingual[df_bilingual["source"] == "flores_dev"]

    logger.info(f"\n{'=' * 55}\n📊 RAPPORT FINAL\n{'=' * 55}")
    logger.info(f"Bilingue total    : {len(df_bilingual):,} paires sauvegardées dans {output_file}")
    logger.info(f"  Entraînement    : {len(train_df):,}")
    logger.info(f"  Validation      : {len(valid_df):,} (FLORES dev)")

    if len(train_df) > 0:
        logger.info("\nRépartition par source (dataset) :")
        logger.info(df_bilingual.groupby("source").size().to_string())


if __name__ == "__main__":
    build_corpus()