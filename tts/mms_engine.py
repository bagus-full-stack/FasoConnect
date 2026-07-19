# tts/mms_engine.py
import torch
import numpy as np
import io
import base64
import logging
import scipy.io.wavfile as wavfile
from dataclasses import dataclass
from transformers import VitsModel, AutoTokenizer

# Mapping langue → modèle MMS-TTS (huggingface.co/facebook/mms-tts-xxx)
MMS_TTS_MODELS = {
    "moore":        "facebook/mms-tts-mos",
    "dioula":       "facebook/mms-tts-dyu",
    "fulfulde":     "facebook/mms-tts-fuv",
    "gourmantsema": "facebook/mms-tts-gux",
    "dagaare":      "facebook/mms-tts-dga",
    "francais":     "facebook/mms-tts-fra",
    "anglais":      "facebook/mms-tts-eng",
}


@dataclass
class TTSResult:
    audio_b64: str          # audio WAV encodé en base64
    sample_rate: int
    duration_seconds: float


class MMSTTSEngine:
    """
    Moteur Text-To-Speech basé sur Meta MMS.
    Charge les modèles à la demande et les garde en cache mémoire.
    """

    def __init__(self):
        self._model_cache: dict = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"MMS-TTS initialisé sur {self.device}")

    def _load_model(self, lang_key: str):
        """Charge le modèle TTS pour une langue donnée (avec cache mémoire)."""
        if lang_key not in self._model_cache:
            model_id = MMS_TTS_MODELS.get(lang_key)
            if not model_id:
                raise ValueError(f"Langue non supportée pour TTS : '{lang_key}'. "
                                 f"Langues disponibles : {list(MMS_TTS_MODELS.keys())}")

            logging.info(f"Chargement MMS-TTS pour '{lang_key}' ({model_id})...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id).to(self.device)
            model.eval()
            self._model_cache[lang_key] = (tokenizer, model)
            logging.info(f"✅ MMS-TTS '{lang_key}' prêt")

        return self._model_cache[lang_key]

    def synthesize(self, text: str, lang: str, speed: float = 1.0) -> TTSResult:
        """
        Génère un fichier audio WAV encodé en base64 à partir d'un texte.

        Args:
            text  : texte à synthétiser
            lang  : clé de MMS_TTS_MODELS (ex: 'moore', 'francais')
            speed : vitesse de lecture (0.5 = lent, 1.0 = normal, 2.0 = rapide)

        Returns:
            TTSResult avec audio_b64, sample_rate, duration_seconds
        """
        tokenizer, model = self._load_model(lang)

        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = model(**inputs)

        # Waveform : shape [1, T] → flatten en [T]
        waveform = output.waveform.squeeze().cpu().numpy()
        sample_rate = model.config.sampling_rate  # généralement 16000 Hz

        # Ajustement de la vitesse par rééchantillonnage
        if speed != 1.0:
            import scipy.signal as signal
            target_len = int(len(waveform) / speed)
            waveform = signal.resample(waveform, target_len)

        duration = round(len(waveform) / sample_rate, 2)

        # Encodage WAV → base64
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, (waveform * 32767).astype(np.int16))
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()

        return TTSResult(
            audio_b64=audio_b64,
            sample_rate=sample_rate,
            duration_seconds=duration,
        )

    def preload(self, langs: list[str]):
        """Précharge les modèles TTS pour une liste de langues au démarrage."""
        for lang in langs:
            try:
                self._load_model(lang)
            except ValueError as e:
                logging.warning(f"Préchargement ignoré : {e}")