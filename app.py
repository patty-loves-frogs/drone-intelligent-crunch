import os
import io
import json
import base64
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

# Charger les variables d'environnement TÔT pour qu'elles soient disponibles lors des imports suivants !
load_dotenv()

import pandas as pd
import streamlit as st
import markdown

from graph import yolo_node, vlm_node
from reporter import reporter_node

try:
    from weasyprint import HTML
except ImportError:
    HTML = None

# ============================================================
# GESTION DES CONVERSATIONS & CONFIGURATION
# ============================================================
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)
MODELS_CONFIG_FILE = Path("models_config.json")

@st.cache_resource
def load_models_config():
    if MODELS_CONFIG_FILE.exists():
        try:
            with open(MODELS_CONFIG_FILE, 'r', encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_models_config(vlm, reporter):
    with open(MODELS_CONFIG_FILE, 'w', encoding="utf-8") as f:
        json.dump({"vlm_model": vlm, "reporter_model": reporter}, f, indent=4)
        
    # On met à jour le cache manuellement (Streamlit < 1.30 : st.cache_resource.clear() )
    if hasattr(load_models_config, "clear"):
         load_models_config.clear()

def save_conversation():
    if st.session_state.conversation_id and st.session_state.conversation_history:
        conv_file = CONVERSATIONS_DIR / f"{st.session_state.conversation_id}.json"
        data = {
            "id": st.session_state.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.conversation_history,
        }
        with open(conv_file, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

def load_conversation(conversation_id: str):
    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if conv_file.exists():
        with open(conv_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        st.session_state.conversation_id = conversation_id
        st.session_state.conversation_history = data.get("messages", [])
        st.rerun()

def new_conversation():
    save_conversation()  # Sauvegarde avant de réinitialiser
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.conversation_history = []
    st.rerun()

def get_conversation_list() -> list:
    conversations = []
    for conv_file in sorted(CONVERSATIONS_DIR.glob("*.json"), reverse=True):
        try:
            with open(conv_file, 'r', encoding="utf-8") as f:
                data = json.load(f)
            first_msg = data.get("messages", [{}])[0].get("content", "Nouvelle discussion")[:30]
            conversations.append({
                "id": data.get("id"),
                "timestamp": data.get("timestamp", ""),
                "preview": f"{first_msg}..."
            })
        except Exception:
            pass
    return conversations

def get_available_ollama_models(base_url: str):
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
    except Exception:
        pass
    return ["llava:latest", "qwen2.5:3b"]


# Configuration de la page
st.set_page_config(page_title="Drone Intelligent Conversationnel", layout="wide")

# CSS hérité pour le style
st.markdown("""
<style>
    :root {
        --primary: #00d4aa;
        --secondary: #0099ff;
        --accent: #ff6b6b;
    }
    h1 {
        background: linear-gradient(135deg, #00d4aa 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: rgba(255, 255, 255, 0.7);
        border-left: 4px solid #00d4aa;
    }
</style>
""", unsafe_allow_html=True)

st.title("Drone Intelligent Conversationnel")

# Dossiers de stockage
UPLOADED_VIDEOS_DIR = Path("uploaded_videos")
UPLOADED_VIDEOS_DIR.mkdir(exist_ok=True)

# Initialisation des variables de session
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "final_state" not in st.session_state:
    st.session_state.final_state = None
if "ollama_base_url" not in st.session_state:
    st.session_state.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
if "available_models" not in st.session_state:
    st.session_state.available_models = get_available_ollama_models(st.session_state.ollama_base_url)

models_config = load_models_config()

if "selected_vlm_model" not in st.session_state:
    default_vlm = models_config.get("vlm_model", "llava:latest")
    st.session_state.selected_vlm_model = default_vlm if default_vlm in st.session_state.available_models else (st.session_state.available_models[0] if st.session_state.available_models else "llava:latest")
if "selected_reporter_model" not in st.session_state:
    default_rep = models_config.get("reporter_model", "qwen2.5:3b")
    st.session_state.selected_reporter_model = default_rep if default_rep in st.session_state.available_models else (st.session_state.available_models[0] if st.session_state.available_models else "qwen2.5:3b")

# Assigner l'environnement depuis la session pour que les scripts backend en héritent dynamiquement
os.environ["OLLAMA_BASE_URL"] = st.session_state.ollama_base_url
os.environ["OLLAMA_MODEL"] = st.session_state.selected_vlm_model
os.environ["REPORTER_MODEL"] = st.session_state.selected_reporter_model


def add_message(role: str, content: str):
    st.session_state.conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    save_conversation()


# Barre latérale (Sidebar) pour l'upload et rapport
with st.sidebar:
    st.markdown("### Configuration Modèles")
    
    selected_vlm = st.selectbox("Modèle VLM", st.session_state.available_models, index=st.session_state.available_models.index(st.session_state.selected_vlm_model) if st.session_state.selected_vlm_model in st.session_state.available_models else 0)
    if selected_vlm != st.session_state.selected_vlm_model:
        st.session_state.selected_vlm_model = selected_vlm
        os.environ["OLLAMA_MODEL"] = selected_vlm
        save_models_config(st.session_state.selected_vlm_model, st.session_state.selected_reporter_model)
        st.rerun()

    selected_reporter = st.selectbox("Modèle Reporter", st.session_state.available_models, index=st.session_state.available_models.index(st.session_state.selected_reporter_model) if st.session_state.selected_reporter_model in st.session_state.available_models else 0)
    if selected_reporter != st.session_state.selected_reporter_model:
        st.session_state.selected_reporter_model = selected_reporter
        os.environ["REPORTER_MODEL"] = selected_reporter
        save_models_config(st.session_state.selected_vlm_model, st.session_state.selected_reporter_model)
        st.rerun()

    st.markdown("---")
    st.markdown("### Médias")
    
    uploaded_file = st.file_uploader("Upload Vidéo", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        video_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
        video_path = UPLOADED_VIDEOS_DIR / video_filename
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.read())
        st.session_state.uploaded_video_path = str(video_path)
        st.success(f"Vidéo chargée : {uploaded_file.name}")

    st.markdown("---")
    st.markdown("### Historique Discussions")
    
    if st.button("Nouvelle conversation", use_container_width=True):
        new_conversation()
        
    conversations = get_conversation_list()
    for conv in conversations:
        if st.button(conv["preview"], key=conv["id"], use_container_width=True):
            load_conversation(conv["id"])


# Zone de conversation principale
message_container = st.container()

with message_container:
    if not st.session_state.conversation_history:
        st.info("Bonjour ! Uploadez une vidéo dans la barre latérale, puis donnez-moi une instruction ci-dessous.")
        
    for entry in st.session_state.conversation_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"], unsafe_allow_html=True)
            
            # Afficher des détails si c'est la dernière réponse et qu'on a un état final
            if entry["role"] == "assistant" and "final_state" in entry and entry["final_state"]:
                fs = entry["final_state"]
                images = fs.get("images", [])
                if images:
                    st.write("**Preuves visuelles :**")
                    cols = st.columns(min(3, len(images)))
                    for idx, img_path in enumerate(images):
                        if os.path.isfile(img_path):
                            with cols[idx % 3]:
                                st.image(img_path, use_container_width=True)


# Entrée utilisateur
user_input = st.chat_input("Ex: 'Cherche des personnes allongées dans la vidéo'")

if user_input:
    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        if not st.session_state.uploaded_video_path:
            response = "⚠️ Veuillez télécharger une vidéo dans le panneau latéral avant de lancer l'analyse."
            st.warning(response)
            add_message("assistant", response)
        else:
            progress = st.progress(0)
            status_container = st.empty()
            
            initial_state = {
                "instruction": user_input,
                "video_path": st.session_state.uploaded_video_path,
                "images": [],
                "detections": [],
                "vlm_candidates": [],
                "vlm_observations": [],
                "report": "",
                "yolo_json": {},
            }
            
            try:
                status_container.info("Étape 1/3 : Analyse vidéo avec YOLO en cours...")
                progress.progress(15)
                state = yolo_node(initial_state)
                
                status_container.success("YOLO terminé.")
                progress.progress(50)
                
                status_container.info("Étape 2/3 : Analyse VLM des images critiques en cours...")
                progress.progress(60)
                state = vlm_node(state)
                
                status_container.success("VLM terminé.")
                progress.progress(80)
                
                status_container.info("Étape 3/3 : Génération du rapport de mission...")
                progress.progress(90)
                final_state = reporter_node(state)
                
                status_container.success("Analyse terminée.")
                progress.progress(100)
                status_container.empty()
                progress.empty()
                
                report = final_state.get("report", "Analyse terminée sans rapport généré.")
                st.session_state.final_state = final_state
                
                # HTML du rapport pour l'embed local
                html_report_content = markdown.markdown(report)
                full_html = f"<html><head><meta charset='utf-8'></head><body style='font-family: sans-serif; padding: 20px;'><h1>Rapport Drone</h1><div>{html_report_content}</div></body></html>"
                
                pdf_bytes = None
                if HTML:
                    try:
                        pdf_bytes = HTML(string=full_html).write_pdf()
                    except Exception:
                        pass
                
                if pdf_bytes:
                    b64 = base64.b64encode(pdf_bytes).decode()
                    download_href = f'<a href="data:application/pdf;base64,{b64}" download="rapport.pdf" target="_blank" type="application/pdf" style="display:inline-block;padding:10px 15px;background:#00d4aa;color:white;border-radius:8px;text-decoration:none;">📄 Télécharger le Rapport (PDF)</a>'
                else:
                    b64 = base64.b64encode(full_html.encode('utf-8')).decode()
                    download_href = f'<a href="data:text/html;base64,{b64}" download="rapport.html" target="_blank" type="text/html" style="display:inline-block;padding:10px 15px;background:#00d4aa;color:white;border-radius:8px;text-decoration:none;">📄 Télécharger le Rapport (HTML)</a>'
                
                bot_reply = f"✅ L'analyse est terminée.\n\n{download_href}"
                st.markdown(bot_reply, unsafe_allow_html=True)
                
                images = final_state.get("images", [])
                if images:
                    st.write("**Preuves visuelles :**")
                    cols = st.columns(3)
                    for idx, img_path in enumerate(images):
                        if os.path.isfile(img_path):
                            with cols[idx % 3]:
                                st.image(img_path, use_container_width=True)
                                
                # On sauvegarde l'état avec le message pour pouvoir réafficher les images dans l'historique
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": bot_reply,
                    "timestamp": datetime.now().isoformat(),
                    "final_state": final_state
                })
                save_conversation()
                
            except Exception as e:
                error_msg = f"Erreur lors de l'exécution: {str(e)}"
                status_container.empty()
                if 'progress' in locals():
                    progress.empty()
                st.error(error_msg)
                add_message("assistant", error_msg)
