<p align="center">
  <a href="README.md">English</a>
  &nbsp;·&nbsp;
  <a href="README.fr.md"><b>Français</b></a>
  &nbsp;·&nbsp;
  <a href="README.es.md">Español</a>
</p>

Clonage de voix avec Coqui XTTS v2. `app.py` charge `tts_models/multilingual/multi-dataset/xtts_v2` sur CUDA, lit la référence `tts.wav` et écrit `sortie.wav` à partir de la phrase française du fichier.

```bash
pip install -r requirements.txt
python app.py
```

Il n’y a pas de README. Modifiez le texte, le fichier locuteur ou la langue dans `app.py`. Le script exige un GPU CUDA.
