import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    print("✅ HuggingFace authentifié")
else:
    print("⚠️  HF_TOKEN manquant")

print("Téléchargement NLLB-200...")
AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-1.3B",
    low_cpu_mem_usage=True,
)
print("✅ Modèles téléchargés.")