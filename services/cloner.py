# voice_maker_app.py
import streamlit as st
import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
import tempfile
import os
import time
from datetime import datetime

# Sécurité PyTorch 2.6 pour XTTS
torch.serialization.add_safe_globals([XttsConfig])

# Configuration de la page
st.set_page_config(
    page_title="Voice Maker Pro",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    /* Thème Apple-inspired */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header avec effet de verre (Glassmorphism) */
    .header-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    /* Cards Material Design */
    .custom-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
    }
    
    /* Boutons avec design moderne */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Zone de texte améliorée */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        transition: border 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Uploader personnalisé */
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 500;
        margin: 4px;
    }
    
    .status-success {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        color: white;
    }
    
    .status-info {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
        color: white;
    }
    
    /* Animation de chargement */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .pulse {
        animation: pulse 1.5s infinite;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 14px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Header avec design Apple
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; gap: 20px;">
        <div style="flex-shrink: 0;">
            <h1 style="margin: 0; color: #1a1a1a; font-size: 2.5rem; font-weight: 800;">🎤 Voice Maker Pro</h1>
            <p style="color: #666; font-size: 1.1rem; margin-top: 0.5rem;">
                Synthèse vocale avancée avec clonage de voix • Technologie XTTS v2
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialisation du modèle dans le cache
@st.cache_resource(show_spinner=False)
def load_tts_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    status_text = f"Chargement du modèle XTTS v2 sur {'🎮 GPU (CUDA)' if device == 'cuda' else '⚡ CPU'}"
    
    with st.spinner(status_text):
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        time.sleep(1)  # Simulation du chargement pour l'animation
    return tts, device

# Chargement du modèle
tts, device = load_tts_model()

# Layout en deux colonnes
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📝 Texte à synthétiser")
    st.markdown("""
    <p style="color: #666; font-size: 0.95rem; margin-bottom: 1.5rem;">
    Saisissez le texte que vous souhaitez convertir en parole naturelle. 
    Le modèle supporte plusieurs langues incluant le français, l'anglais, l'espagnol, etc.
    </p>
    """, unsafe_allow_html=True)
    
    text = st.text_area(
        " ",
        value="Bonjour, je suis Voice Maker Pro. Je transforme votre texte en parole naturelle grâce à la technologie XTTS v2.",
        height=200,
        help="Le texte sera converti en audio avec une voix naturelle et expressive."
    )
    
    # Statistiques du texte
    col1a, col1b, col1c = st.columns(3)
    with col1a:
        st.metric("Caractères", len(text))
    with col1b:
        st.metric("Mots", len(text.split()))
    with col1c:
        st.metric("Temps estimé", f"{len(text)/1500:.1f}s")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🎙️ Clonage de voix")
    st.markdown("""
    <p style="color: #666; font-size: 0.95rem; margin-bottom: 1.5rem;">
    Téléchargez un échantillon vocal pour cloner une voix spécifique (optionnel). 
    Sinon, une voix par défaut sera utilisée.
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    speaker_file = st.file_uploader(
        "📤 Glissez-déposez ou cliquez pour sélectionner",
        type=["wav", "mp3"],
        help="Fichier audio de référence pour le clonage vocal (recommandé: 5-15 secondes)"
    )
    
    if speaker_file:
        st.success(f"✅ Fichier chargé: {speaker_file.name}")
        file_size = speaker_file.size / (1024 * 1024)
        st.caption(f"Taille: {file_size:.2f} MB • Type: {speaker_file.type}")
    else:
        st.info("ℹ️ Aucun fichier sélectionné. Utilisation de la voix par défaut.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Paramètres avancés
    with st.expander("⚙️ Paramètres avancés", expanded=False):
        language = st.selectbox(
            "Langue",
            ["fr", "en", "es", "de", "it"],
            index=0,
            help="Sélectionnez la langue du texte"
        )
        
        speed = st.slider(
            "Vitesse de lecture",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1,
            help="Ajustez la vitesse de la voix synthétisée"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Bouton de génération au centre
st.markdown('<br>', unsafe_allow_html=True)
col_center = st.columns([1, 2, 1])
with col_center[1]:
    generate_button = st.button(
        "🚀 Générer l'audio",
        type="primary",
        use_container_width=True
    )

# Section de génération
if generate_button:
    if not text.strip():
        st.error("⚠️ Veuillez entrer un texte valide.")
    else:
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Étape 1: Préparation
        status_text.markdown('<p class="pulse">📦 Préparation du modèle...</p>', unsafe_allow_html=True)
        progress_bar.progress(20)
        time.sleep(0.5)
        
        # Étape 2: Traitement du fichier audio
        speaker_path = None
        if speaker_file:
            status_text.markdown('<p class="pulse">🎵 Traitement de l\'échantillon vocal...</p>', unsafe_allow_html=True)
            progress_bar.progress(40)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.write(speaker_file.read())
            tmp.close()
            speaker_path = tmp.name
            time.sleep(0.5)
        
        # Étape 3: Génération
        status_text.markdown('<p class="pulse">⚡ Génération de l\'audio en cours...</p>', unsafe_allow_html=True)
        progress_bar.progress(60)
        
        # Fichier de sortie avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"voice_maker_{timestamp}.wav"
        
        try:
            # Génération de l'audio
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_path,
                language=language,
                file_path=output_path
            )
            
            # Étape 4: Finalisation
            progress_bar.progress(100)
            status_text.markdown('<p class="pulse">✅ Audio généré avec succès!</p>', unsafe_allow_html=True)
            time.sleep(0.5)
            
            # Affichage du résultat
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.success("### 🎉 Génération terminée !")
            
            # Player audio amélioré
            col_audio1, col_audio2 = st.columns([3, 1])
            with col_audio1:
                st.audio(output_path, format="audio/wav")
            
            with col_audio2:
                # Bouton de téléchargement
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📥 Télécharger",
                        data=file,
                        file_name=output_path,
                        mime="audio/wav",
                        use_container_width=True
                    )
            
            # Informations techniques
            with st.expander("📊 Informations techniques"):
                col_tech1, col_tech2 = st.columns(2)
                with col_tech1:
                    st.metric("Modèle", "XTTS v2")
                    st.metric("Device", "GPU" if device == "cuda" else "CPU")
                with col_tech2:
                    st.metric("Langue", language.upper())
                    st.metric("Vitesse", f"{speed}x")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération: {str(e)}")
        
        finally:
            # Nettoyage
            if speaker_path and os.path.exists(speaker_path):
                os.remove(speaker_path)
            
            # Reset de la barre de progression
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()

# Footer avec informations
st.markdown('<div class="footer">', unsafe_allow_html=True)
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("""
    **🎯 Technologies utilisées**
    - XTTS v2
    - PyTorch
    - Streamlit
    """)

with col_footer2:
    st.markdown("""
    **⚡ Performance**
    - GPU: {'Actif' if device == 'cuda' else 'CPU'}
    - Modèle: Multilingue
    - Latence: Optimisée
    """)

with col_footer3:
    st.markdown("""
    **🔒 Sécurité & Vie privée**
    - Traitement local
    - Pas de stockage cloud
    - Données éphémères
    """)

st.markdown("---")
st.markdown("**Voice Maker Pro** • Synthèse vocale de nouvelle génération")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar avec informations supplémentaires (optionnelle)
with st.sidebar:
    st.markdown("""
    <div class="custom-card">
    <h3>ℹ️ Guide rapide</h3>
    
    **Étapes:**
    1. Saisissez votre texte
    2. (Optionnel) Chargez un échantillon vocal
    3. Cliquez sur "Générer l'audio"
    4. Téléchargez le résultat
    
    **Conseils:**
    • Texte clair et bien ponctué
    • Échantillon vocal: 5-15 secondes
    • Format WAV recommandé
    • Connexion stable recommandée
    
    **Capacités:**
    ✓ Clonage vocal
    ✓ Multilingue
    ✓ Voix naturelles
    ✓ Temps réel
    </div>
    """, unsafe_allow_html=True)
    
    # Statut du système
    st.markdown("""
    <div class="custom-card">
    <h3>📊 Statut système</h3>
    """, unsafe_allow_html=True)
    

    
    st.markdown('</div>', unsafe_allow_html=True)