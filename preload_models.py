# preload_models.py
# Télécharge les modèles NLLB-200 au moment du docker build
# pour éviter le téléchargement au premier appel API.
import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    print("✅ HuggingFace authentifié")
else:
    print("⚠️  HF_TOKEN manquant — téléchargement non authentifié")

print("Téléchargement NLLB-200 distilled 600M...")
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    low_cpu_mem_usage=True,
)
print("✅ Modèles téléchargés.")