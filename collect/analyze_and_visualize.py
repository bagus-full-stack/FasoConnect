# collect/analyze_and_visualize.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# Désactiver les avertissements liés aux regex de Pandas
warnings.filterwarnings("ignore", category=UserWarning)

CORPUS_PATH = "data/processed/corpus_burkina_clean.csv"
OUTPUT_IMG = "data/processed/analyse_dashboard.png"


def run_analysis():
    print(f"Chargement de {CORPUS_PATH}...\n")
    if not Path(CORPUS_PATH).exists():
        print(f"❌ Erreur : Le fichier {CORPUS_PATH} est introuvable.")
        return

    df = pd.read_csv(CORPUS_PATH)

    # Pré-calcul des longueurs pour réutilisation
    df['src_len'] = df['src'].astype(str).apply(lambda x: len(x.split()))
    df['tgt_len'] = df['tgt'].astype(str).apply(lambda x: len(x.split()))

    # ==========================================
    # PARTIE 1 : ANALYSE TEXTUELLE (CONSOLE)
    # ==========================================
    print("=" * 50)
    print("📊 1. APERÇU DU DATASET & DISTRIBUTION")
    print("=" * 50)
    print(f"Total des paires nettoyées : {len(df):,}\n")
    print("Répartition par langue cible :")
    print(df['tgt_lang'].value_counts().to_string())

    if 'source' in df.columns:
        print("\nRépartition par source d'origine :")
        print(df['source'].value_counts().to_string())

    print("\n" + "=" * 50)
    print("📏 2. ANALYSE DES LONGUEURS (en mots)")
    print("=" * 50)
    print(f"Moyenne Source (FR/EN) : {round(df['src_len'].mean(), 1)} mots (Max: {df['src_len'].max()})")
    print("\nStatistiques Cible par Langue :")
    stats = df.groupby('tgt_lang')['tgt_len'].describe(percentiles=[0.5, 0.95])
    print(stats[['count', 'mean', '50%', '95%']].round(1).to_string())

    print("\n" + "=" * 50)
    print("🌾 3. PRÉSENCE DU DOMAINE AGRO/SANTÉ")
    print("=" * 50)
    keywords = r"(?i)(malad|champ|semence|récolt|plante|feuill|insect|traitement|symptôm|eau|pluie|terre|virus|santé|agriculture|élevage|bétail|engrais|pesticid|sécheresse)"
    agro_count = df['src'].astype(str).str.contains(keywords, regex=True).sum()
    print(f"Phrases contenant du vocabulaire Agro/Santé : {agro_count:,} ({agro_count / len(df) * 100:.1f}%)\n")

    print("=" * 50)
    print("⚠️ 4. CHEVAUCHEMENT (Risque de Data Leakage)")
    print("=" * 50)

    def exact_overlap(row):
        src_words = set(str(row['src']).lower().split())
        tgt_words = set(str(row['tgt']).lower().split())
        if not src_words: return 0
        return len(src_words.intersection(tgt_words)) / len(src_words)

    df['overlap_ratio'] = df.apply(exact_overlap, axis=1)
    high_overlap = df[df['overlap_ratio'] > 0.5]
    print(
        f"Phrases suspectes (>50% de mots identiques) : {len(high_overlap):,} ({len(high_overlap) / len(df) * 100:.2f}%)\n")

    print("=" * 50)
    print("🔤 5. CARACTÈRES SPÉCIAUX (CIBLE)")
    print("=" * 50)
    all_tgt_text = "".join(df['tgt'].astype(str).sample(min(10000, len(df))).tolist())
    unique_chars = sorted(list(set(all_tgt_text)))
    special_chars = [c for c in unique_chars if not (c.isascii() and c.isalnum()) and c != ' ']
    print(f"Caractères trouvés : {' '.join(special_chars[:50])}")
    if len(special_chars) > 50:
        print("... (liste tronquée)")

    # ==========================================
    # PARTIE 2 : VISUALISATION (GRAPHIQUES)
    # ==========================================
    print("\n" + "=" * 50)
    print("📈 6. GÉNÉRATION DU TABLEAU DE BORD (DASHBOARD)")
    print("=" * 50)
    print("Création des graphiques en cours...")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Analyse du Corpus AgroSahel (NLLB Fine-tuning)", fontsize=18, fontweight='bold')

    # 1. Langues (Bar Chart)
    sns.countplot(data=df, y='tgt_lang', order=df['tgt_lang'].value_counts().index, palette='viridis', ax=axes[0, 0])
    axes[0, 0].set_title('Répartition par Langue Cible', fontsize=14)
    axes[0, 0].set_xlabel('Nombre de phrases')
    axes[0, 0].set_ylabel('Langue')

    # 2. Sources (Bar Chart)
    if 'source' in df.columns:
        sns.countplot(data=df, y='source', order=df['source'].value_counts().index, palette='magma', ax=axes[0, 1])
        axes[0, 1].set_title('Origine des Données (Sources)', fontsize=14)
        axes[0, 1].set_xlabel('Nombre de phrases')
        axes[0, 1].set_ylabel('Dataset')

    # 3. Longueurs (Boxplot)
    sns.boxplot(data=df, x='tgt_len', y='tgt_lang', palette='Set2', ax=axes[1, 0], showfliers=False)
    axes[1, 0].set_title('Distribution des Longueurs (sans valeurs extrêmes)', fontsize=14)
    axes[1, 0].set_xlabel('Nombre de Mots')
    axes[1, 0].set_ylabel('')

    # 4. Domaine (Pie Chart)
    other_count = len(df) - agro_count
    axes[1, 1].pie(
        [agro_count, other_count],
        labels=['Domaine Agro/Santé', 'Langage Général'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#bdc3c7'],
        explode=(0.1, 0),
        shadow=True,
        startangle=90
    )
    axes[1, 1].set_title('Proportion du Vocabulaire Métier', fontsize=14)

    # Sauvegarde de l'image
    OUTPUT_IMG_PATH = Path(OUTPUT_IMG)
    OUTPUT_IMG_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_IMG_PATH, dpi=300)
    plt.close()

    print(f"✅ Tableau de bord généré avec succès : {OUTPUT_IMG_PATH}")


if __name__ == "__main__":
    run_analysis()