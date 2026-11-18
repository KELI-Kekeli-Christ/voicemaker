<p align="center">
  <a href="README.md">English</a>
  &nbsp;·&nbsp;
  <a href="README.fr.md">Français</a>
  &nbsp;·&nbsp;
  <a href="README.es.md"><b>Español</b></a>
</p>

Clonación de voz con Coqui XTTS v2. `app.py` carga `tts_models/multilingual/multi-dataset/xtts_v2` en CUDA, lee la referencia `tts.wav` y escribe `sortie.wav` a partir de la frase francesa del archivo.

```bash
pip install -r requirements.txt
python app.py
```

No hay README. Cambia el texto, el archivo del locutor o el idioma dentro de `app.py`. El script necesita un dispositivo CUDA.
