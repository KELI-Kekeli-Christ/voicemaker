<p align="center">
  <a href="README.md"><b>English</b></a>
  &nbsp;·&nbsp;
  <a href="README.fr.md">Français</a>
  &nbsp;·&nbsp;
  <a href="README.es.md">Español</a>
</p>

Voice cloning with Coqui XTTS v2. `app.py` loads `tts_models/multilingual/multi-dataset/xtts_v2` on CUDA, reads the reference `tts.wav`, and writes `sortie.wav` from the French sentence in the file.

```bash
pip install -r requirements.txt
python app.py
```

There is no README. Change the text, the speaker file, or the language inside `app.py`. The script requires a CUDA device.
