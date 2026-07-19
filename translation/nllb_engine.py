# translation/nllb_engine.py
import torch
import logging
from typing import Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Codes de langue NLLB-200 pour les langues burkinabè
BURKINA_LANG_CODES = {
    "moore":        "mos_Latn",
    "dioula":       "dyu_Latn",
    "fulfulde":     "fuv_Latn",
    "gourmantsema": "gux_Latn",
    "dagaare":      "dga_Latn",
    "francais":     "fra_Latn",
    "anglais":      "eng_Latn",
}


class NLLBTranslator:
    """
    Traducteur multilingue basé sur Meta NLLB-200.
    Supporte toutes les paires bidirectionnelles entre
    langues burkinabè, français et anglais.
    """

    # facebook/nllb-200-distilled-600M  → léger, CPU-friendly (~3 Go RAM)
    # facebook/nllb-200-distilled-1.3B  → bon compromis  (~5 Go RAM / ~2.6 Go VRAM)
    # facebook/nllb-200-3.3B            → meilleure qualité (~12 Go RAM)
    MODEL_ID = "facebook/nllb-200-distilled-1.3B"

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Chargement NLLB sur {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_ID,
            # ✅ dtype au lieu de torch_dtype (dépréciation corrigée)
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            # ✅ charge progressivement pour éviter les pics RAM
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.model.eval()
        logging.info(f"✅ NLLB-200 chargé sur {self.device}")

    def _resolve_lang_id(self, lang_code: str) -> int:
        """
        Résout l'ID du token de langue de façon compatible
        avec toutes les versions de transformers.
        ✅ Fix : convert_tokens_to_ids remplace lang_code_to_id (déprécié)
        """
        token_id = self.tokenizer.convert_tokens_to_ids(lang_code)
        if token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Code langue non reconnu par NLLB : '{lang_code}'. "
                f"Codes disponibles : {list(BURKINA_LANG_CODES.values())}"
            )
        return token_id

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 512,
        num_beams: int = 5,
    ) -> str:
        """Traduit un texte d'une langue vers une autre."""
        src_code = BURKINA_LANG_CODES.get(src_lang, src_lang)
        tgt_code = BURKINA_LANG_CODES.get(tgt_lang, tgt_lang)

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        tgt_lang_id = self._resolve_lang_id(tgt_code)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        return self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    def translate_batch(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
    ) -> list[str]:
        """Traduction par lot pour optimiser le débit."""
        src_code = BURKINA_LANG_CODES.get(src_lang, src_lang)
        tgt_code = BURKINA_LANG_CODES.get(tgt_lang, tgt_lang)

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        tgt_lang_id = self._resolve_lang_id(tgt_code)

        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_length=512,
                num_beams=4,
            )

        return [
            self.tokenizer.decode(tok, skip_special_tokens=True)
            for tok in output_tokens
        ]