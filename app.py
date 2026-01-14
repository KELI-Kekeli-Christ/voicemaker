import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig

# Allowlist the XTTS config class for PyTorch 2.6 security
torch.serialization.add_safe_globals([XttsConfig])

# Now initialize the model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

tts.tts_to_file(
    text="Bonjour, ma voix clonée fonctionne parfaitement avec XTTS.",
    speaker_wav="tts.wav",
    language="fr",
    file_path="sortie.wav"
)

print("Audio généré : sortie.wav")