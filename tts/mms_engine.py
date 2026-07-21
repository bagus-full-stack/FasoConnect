# tts/mms_engine.py — VERSION FINALE
import torch
import numpy as np
import io
import base64
import logging
import scipy.io.wavfile as wavfile
from dataclasses import dataclass
from transformers import VitsModel, AutoTokenizer

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
    audio_b64:        str
    sample_rate:      int
    duration_seconds: float


class MMSTTSEngine:
    """
    Moteur TTS Meta MMS.
    Modèles chargés à la demande et mis en cache mémoire.
    L'audio n'est jamais stocké sur disque — base64 uniquement.
    """

    def __init__(self):
        self._cache: dict = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"MMS-TTS initialisé sur {self.device}")

    def _load_model(self, lang: str):
        if lang not in self._cache:
            model_id = MMS_TTS_MODELS.get(lang)
            if not model_id:
                raise ValueError(
                    f"Langue TTS non supportée : '{lang}'. "
                    f"Disponibles : {list(MMS_TTS_MODELS.keys())}"
                )
            logging.info(f"Chargement MMS-TTS '{lang}'...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id).to(self.device)
            model.eval()
            self._cache[lang] = (tokenizer, model)
            logging.info(f"✅ MMS-TTS '{lang}' prêt")
        return self._cache[lang]

    def synthesize(self, text: str, lang: str, speed: float = 1.0) -> TTSResult:
        tokenizer, model = self._load_model(lang)

        inputs = tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = model(**inputs)

        waveform    = output.waveform.squeeze().cpu().numpy()
        sample_rate = model.config.sampling_rate

        if speed != 1.0:
            import scipy.signal as signal
            waveform = signal.resample(waveform, int(len(waveform) / speed))

        duration = round(len(waveform) / sample_rate, 2)

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, (waveform * 32767).astype(np.int16))
        audio_b64 = base64.b64encode(buffer.getvalue()).decode()

        return TTSResult(
            audio_b64=audio_b64,
            sample_rate=sample_rate,
            duration_seconds=duration,
        )

    def preload(self, langs: list[str]):
        """Préchauffage des modèles TTS au démarrage."""
        for lang in langs:
            try:
                self._load_model(lang)
            except ValueError as e:
                logging.warning(f"Préchargement ignoré : {e}")