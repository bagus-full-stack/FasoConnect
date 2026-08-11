# 1. Cloner votre projet (Remplacer l'URL par la vôtre)
!git clone https://github.com/VOTRE_NOM/FasoConnect.git
%cd FasoConnect

# 2. Installer les dépendances
!pip install -r requirements.txt
!pip install sacrebleu evaluate peft bitsandbytes

# 3. Récupérer le corpus nettoyé depuis votre Google Drive
!mkdir -p data/processed
!cp /content/drive/MyDrive/corpus_burkina_clean.csv data/processed/

# 4. S'authentifier sur Hugging Face (Requis pour télécharger NLLB)
import os
os.environ["HF_TOKEN"] = "VOTRE_TOKEN_HUGGINGFACE"

# 5. ÉVALUER LE MODÈLE DE BASE (Votre référence avant de commencer)
!python training/evaluate_model.py --model facebook/nllb-200-distilled-600M

# 6. Lancer LE SCRIPT PYTHON DE FINE-TUNING (Prendra ~1h30 à 2h sur le T4 de Colab)
!python training/finetune_nllb.py

# 7. Lancer LA COMPARAISON FINALE (Génère le tableau comparatif)
!python training/evaluate_model.py --compare

# 8. SAUVEGARDE TOTALE VERS LE DRIVE (Modèle + Tous les résultats d'évaluation)
!cp -r models/nllb-burkina-v1 /content/drive/MyDrive/
!cp -r logs/eval /content/drive/MyDrive/


# Garder la session toujours active

function keepAlive() {
    console.log("Maintien de la connexion Colab en cours...");
    // Simule un clic sur le bouton de connexion en haut à droite
    const connectButton = document.querySelector('colab-connect-button');
    if (connectButton && connectButton.shadowRoot) {
        const btn = connectButton.shadowRoot.getElementById('connect');
        if (btn) btn.click();
    }
}
// Exécute ce faux clic toutes les 60 secondes (60000 millisecondes)
setInterval(keepAlive, 60000);