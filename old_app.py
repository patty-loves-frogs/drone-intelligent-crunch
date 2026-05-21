import os
import json
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from graph import graph
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from agents.yolo import analyze_video_with_yolo
from agents.face_comparison import crop_face_from_image, compare_faces, identify_unique_persons
import base64
from io import BytesIO
import cv2
import numpy as np

try:
    from weasyprint import HTML, CSS
except ImportError:
    HTML = None
    CSS = None

# Load environment variables from .env
load_dotenv()

st.set_page_config(page_title="Drone Intelligent", layout="wide")

# Custom CSS for modern UI with sticky bottom input
st.markdown("""
<style>
    /* ===== COLOR SCHEME: Teal/Green Modern ===== */
    :root {
        --primary: #00d4aa;
        --secondary: #0099ff;
        --accent: #ff6b6b;
        --dark: #1a1a2e;
        --light: #f8f9fa;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e8f4f8 100%);
    }
    
    /* Chat input styling - Teal theme */
    .stChatInput {
        border-radius: 12px !important;
        border: 2px solid #00d4aa !important;
        background: white !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(0, 212, 170, 0.15) !important;
        min-height: 50px !important;
    }
    
    .stChatInput:focus-within {
        border-color: #0099ff !important;
        box-shadow: 0 0 25px rgba(0, 212, 170, 0.5), 0 0 15px rgba(0, 153, 255, 0.3) !important;
        transform: translateY(-2px);
    }
    
    /* Chat input text styling */
    .stChatInput input {
        font-size: 16px !important;
        padding: 12px 16px !important;
    }
    
    /* Button styling - Teal/Green */
    .stButton > button {
        border-radius: 12px !important;
        border: 2px solid #00d4aa !important;
        background: linear-gradient(135deg, #00d4aa 0%, #0099ff 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 18px !important;
        height: 50px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.5) !important;
        background: linear-gradient(135deg, #0099ff 0%, #00d4aa 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Progress bar - Teal */
    .stProgress > div > div > div {
        background: linear-gradient(to right, #00d4aa 0%, #0099ff 100%) !important;
        border-radius: 10px !important;
    }
    
    /* File uploader styling */
    .stFileUploadDropzone {
        border: 2px dashed #00d4aa !important;
        border-radius: 12px;
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.08) 0%, rgba(0, 153, 255, 0.08) 100%) !important;
        transition: all 0.3s ease;
    }
    
    .stFileUploadDropzone:hover {
        border-color: #0099ff !important;
        background: linear-gradient(135deg, rgba(0, 212, 170, 0.15) 0%, rgba(0, 153, 255, 0.15) 100%) !important;
        box-shadow: 0 4px 12px rgba(0, 212, 170, 0.2) !important;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #00d4aa 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: rgba(255, 255, 255, 0.7);
        border-left: 4px solid #00d4aa;
    }
    
    /* Scrollbar styling - Teal */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4aa 0%, #0099ff 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0099ff 0%, #00d4aa 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(0, 212, 170, 0.05) 0%, rgba(0, 153, 255, 0.05) 100%);
    }
    
    /* Toast styling */
    .stToast {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Conversation storage directory
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)

# Uploaded videos directory
UPLOADED_VIDEOS_DIR = Path("uploaded_videos")
UPLOADED_VIDEOS_DIR.mkdir(exist_ok=True)

# Initialize session state
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "video_analysis" not in st.session_state:
    st.session_state.video_analysis = None
if "uploaded_video_path" not in st.session_state:
    st.session_state.uploaded_video_path = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "qwen2.5:3b"
if "available_models" not in st.session_state:
    st.session_state.available_models = []
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
if "ollama_base_url" not in st.session_state:
    # Try to get from environment variable, otherwise default to localhost
    st.session_state.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
if "report_pdf" not in st.session_state:
    st.session_state.report_pdf = None
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False


def save_conversation():
    """Sauvegarde la conversation actuelle."""
    if st.session_state.conversation_id and st.session_state.conversation_history:
        conv_file = CONVERSATIONS_DIR / f"{st.session_state.conversation_id}.json"
        data = {
            "id": st.session_state.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "messages": st.session_state.conversation_history,
            "analysis_results": st.session_state.analysis_results,
        }
        with open(conv_file, 'w') as f:
            json.dump(data, f, indent=2)


def load_conversation(conversation_id: str):
    """Charge une conversation précédente."""
    conv_file = CONVERSATIONS_DIR / f"{conversation_id}.json"
    if conv_file.exists():
        with open(conv_file, 'r') as f:
            data = json.load(f)
        st.session_state.conversation_id = conversation_id
        st.session_state.conversation_history = data.get("messages", [])
        st.session_state.analysis_results = data.get("analysis_results")
        st.rerun()


def new_conversation():
    """Crée une nouvelle conversation."""
    save_conversation()  # Save current one first
    st.session_state.conversation_id = None
    st.session_state.conversation_history = []
    st.session_state.analysis_results = None
    st.rerun()


def get_conversation_list() -> List[Dict]:
    """Retourne la liste des conversations précédentes."""
    conversations = []
    for conv_file in sorted(CONVERSATIONS_DIR.glob("*.json"), reverse=True):
        with open(conv_file, 'r') as f:
            data = json.load(f)
        first_message = data.get("messages", [{}])[0].get("content", "Sans titre")[:50]
        conversations.append({
            "id": data.get("id"),
            "timestamp": data.get("timestamp", ""),
            "preview": first_message,
            "message_count": len(data.get("messages", []))
        })
    return conversations


def get_available_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Récupère les modèles disponibles sur Ollama."""
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            return sorted(list(set(models)))  # Unique and sorted
    except Exception:
        pass
    return ["qwen2.5:3b"]  # Fallback


def get_latest_video_analysis() -> Dict[str, Any] | None:
    """Récupère la dernière analyse vidéo du dossier outputs/runs."""
    output_root = Path("outputs/runs")
    if not output_root.exists():
        return None
    
    # Trouver le dernier dossier créé
    run_dirs = sorted([d for d in output_root.iterdir() if d.is_dir()], 
                      key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not run_dirs:
        return None
    
    latest_run = run_dirs[0]
    analysis_file = latest_run / "analysis.json"
    
    if not analysis_file.exists():
        return None
    
    with open(analysis_file, 'r') as f:
        return json.load(f)


def get_video_summary(analysis: Dict[str, Any]) -> str:
    """Génère un résumé de l'analyse vidéo."""
    if not analysis:
        return "Aucune analyse vidéo disponible."
    
    frames_data = analysis.get("frames", [])
    if not frames_data:
        return "Aucune détection dans l'analyse."
    
    # Compter les détections par risque et posture
    risk_counts = {"ELEVE": 0, "A_VERIFIER": 0, "BAS": 0}
    posture_counts = {"ALLONGE": 0, "DEBOUT": 0, "INCERTAIN": 0}
    
    total_detections = 0
    timestamps = []
    
    for frame in frames_data:
        for detection in frame.get("detections", []):
            total_detections += 1
            risk = detection.get("risk_level", "INCONNU")
            posture = detection.get("posture", "INCONNU")
            
            if risk in risk_counts:
                risk_counts[risk] += 1
            if posture in posture_counts:
                posture_counts[posture] += 1
            
            timestamps.append(frame.get("timestamp_sec", 0))
    
    summary = f"""**Analyse Vidéo - {analysis.get('run_name', 'Inconnue')}**
- Frames analysées: {analysis.get('frames_analyzed', 0)}
- Détections totales: {total_detections}
- Durée: {timestamps[-1]:.2f}s (début) à {timestamps[-1]:.2f}s (fin) si timestamps disponibles

**Par Risque:**
- Élevé: {risk_counts['ELEVE']}
- À vérifier: {risk_counts['A_VERIFIER']}
- Bas: {risk_counts['BAS']}

**Par Posture:**
- Allongé: {posture_counts['ALLONGE']}
- Debout: {posture_counts['DEBOUT']}
- Incertain: {posture_counts['INCERTAIN']}
"""
    return summary


def add_message(role: str, content: str):
    """Ajoute un message à l'historique."""
    # Generate conversation ID on first message if not exists
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.session_state.conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    save_conversation()


def chat_with_llm(user_message: str) -> str:
    """Génère une réponse via Ollama."""
    try:
        analysis = st.session_state.video_analysis or {}
        
        # Build context from conversation history
        context = "Historique de la conversation:\n"
        for msg in st.session_state.conversation_history[-6:]:  # Last 6 messages
            context += f"{msg['role'].upper()}: {msg['content']}\n"
        
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "Tu es un assistant d'analyse vidéo de drone. Tu aides l'utilisateur à comprendre et analyser les détections de personnes "
                "dans les vidéos (posture, position, risque). Réponds toujours en français et sois concis et utile."
            ),
            (
                "human",
                (
                    "{context}\n"
                    "Données d'analyse vidéo disponibles:\n{video_summary}\n\n"
                    "Détails des détections: {detections_detail}\n\n"
                    "Question: {user_message}"
                ),
            ),
        ])
        
        llm = ChatOllama(model=st.session_state.selected_model, base_url=st.session_state.ollama_base_url, temperature=0.3)
        chain = prompt_template | llm
        
        # Détails des détections pour le contexte
        detections_detail = ""
        if analysis and analysis.get("frames"):
            for i, frame in enumerate(analysis["frames"][:10]):  # Limiter à 10 frames
                detections_detail += f"\nFrame {frame.get('frame_index')}, t={frame.get('timestamp_sec')}s:\n"
                for det in frame.get("detections", []):
                    detections_detail += (
                        f"  - Posture: {det.get('posture')}, "
                        f"Risque: {det.get('risk_level')}, "
                        f"Position: {det.get('position_in_image')}, "
                        f"Confiance: {det.get('yolo_confidence', 0):.2f}\n"
                    )
        
        response_obj = chain.invoke({
            "context": context,
            "video_summary": get_video_summary(analysis),
            "detections_detail": detections_detail or "Aucune détection",
            "user_message": user_message,
        })
        
        return response_obj.content.strip()
    
    except Exception as e:
        return f"Erreur Ollama: {str(e)}\n\nAssurez-vous que Ollama est en cours d'exécution avec le modèle {st.session_state.selected_model}"


def process_uploaded_video(video_file) -> Dict[str, Any] | None:
    """Traite une vidéo téléchargée via l'interface."""
    try:
        # Créer un chemin unique pour la vidéo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{timestamp}_{video_file.name}"
        video_path = UPLOADED_VIDEOS_DIR / video_filename
        
        # Sauvegarder la vidéo
        with open(video_path, 'wb') as f:
            f.write(video_file.read())
        
        st.session_state.uploaded_video_path = str(video_path)
        
        # Traiter via YOLO
        analysis_result = analyze_video_with_yolo(str(video_path))
        
        return analysis_result
    except Exception as e:
        st.toast(f"Erreur lors du traitement: {str(e)}", icon="❌")
        return None


def detect_flying_request(message: str) -> bool:
    """Détecte si l'utilisateur demande de voler/scanner une zone."""
    keywords = ["vole", "scan", "détecte", "humain", "survoler", "vidéo", "analyse", "position", "risque", "drone"]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in keywords)


def get_yolo_detection_image() -> tuple[np.ndarray | None, str]:
    """Récupère l'image annotée YOLO montrant les détections de personnes."""
    try:
        if not st.session_state.video_analysis or not st.session_state.video_analysis.get("frames"):
            return None, "Aucune analyse vidéo disponible"
        
        # Chercher une frame avec détections
        frames = st.session_state.video_analysis.get("frames", [])
        
        for frame in frames:
            if frame.get("detections") and len(frame.get("detections", [])) > 0:
                annotated_path = frame.get("annotated_image_path")
                if annotated_path and Path(annotated_path).exists():
                    img = cv2.imread(annotated_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        summary = f"Détections: {len(frame.get('detections', []))} personne(s) - Frame {frame.get('frame_index')}, t={frame.get('timestamp_sec')}s"
                        return img_rgb, summary
        
        return None, "Aucune personne détectée dans la vidéo"
    except Exception as e:
        return None, f"Erreur lors de la lecture d'image: {str(e)}"


def extract_and_display_detected_faces():
    """Extrait les visages détectés par YOLO et les affiche dans le chat."""
    try:
        if not st.session_state.video_analysis or not st.session_state.video_analysis.get("frames"):
            return
        
        frames = st.session_state.video_analysis.get("frames", [])
        all_faces_data = []  # Liste de (face_image, face_path, detection_info)
        
        # Extraire les visages de toutes les détections
        for frame in frames:
            detections = frame.get("detections", [])
            if not detections:
                continue
            
            raw_image_path = frame.get("raw_image_path")
            if not raw_image_path or not Path(raw_image_path).exists():
                continue
            
            for det_idx, detection in enumerate(detections):
                bbox = detection.get("bbox_xyxy")
                if not bbox:
                    continue
                
                face_img, face_path = crop_face_from_image(raw_image_path, bbox)
                
                if face_img is not None:
                    detection_info = {
                        "frame_index": frame.get("frame_index"),
                        "timestamp": frame.get("timestamp_sec"),
                        "posture": detection.get("posture"),
                        "risk": detection.get("risk_level"),
                        "position": detection.get("position_in_image"),
                        "confidence": detection.get("yolo_confidence"),
                        "face_path": face_path
                    }
                    all_faces_data.append((face_img, face_path, detection_info))
        
        if not all_faces_data:
            st.info("Aucun visage détecté à extraire")
            return
        
        # Afficher les faces extraites
        st.markdown("### 👤 Visages Détectés")
        
        # Diviser en colonnes pour afficher plusieurs visages
        cols = st.columns(min(3, len(all_faces_data)))
        
        for idx, (face_img, face_path, det_info) in enumerate(all_faces_data[:9]):  # Max 9 faces
            with cols[idx % 3]:
                st.image(face_img, use_column_width=True)
                st.caption(
                    f"Posture: {det_info['posture']}\n"
                    f"Risque: {det_info['risk']}\n"
                    f"Position: {det_info['position']}\n"
                    f"T: {det_info['timestamp']}s"
                )
        
        # Identifier les personnes uniques par comparaison de visages
        if len(all_faces_data) > 1:
            st.markdown("### 🔍 Analyse des Personnes Uniques")
            
            face_paths = [fp for _, fp, _ in all_faces_data]
            
            with st.spinner("Comparaison des visages..."):
                unique_persons = identify_unique_persons(
                    face_paths,
                    st.session_state.ollama_base_url
                )
            
            # Afficher les résultats
            st.write(f"**Total détections:** {unique_persons['total_detections']}")
            st.write(f"**Personnes uniques identifiées:** {unique_persons['unique_count']}")
            
            # Afficher chaque personne unique
            for person in unique_persons["unique_persons"]:
                person_id = person["person_id"]
                face_indices = person["face_indices"]
                
                with st.expander(f"👤 Personne #{person_id + 1} ({len(face_indices)} détection(s))"):
                    person_cols = st.columns(min(3, len(face_indices)))
                    
                    for col_idx, face_idx in enumerate(face_indices):
                        face_img, face_path, det_info = all_faces_data[face_idx]
                        with person_cols[col_idx % 3]:
                            st.image(face_img, use_column_width=True)
                            st.caption(f"Det #{face_idx + 1} - T:{det_info['timestamp']}s")

    except Exception as e:
        st.error(f"Erreur lors de l'extraction des visages : {str(e)}")


def generate_html_report() -> str:
    """Génère un rapport HTML à partir de la conversation."""
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # Compter les messages
    user_msgs = sum(1 for m in st.session_state.conversation_history if m["role"] == "user")
    assistant_msgs = sum(1 for m in st.session_state.conversation_history if m["role"] == "assistant")
    
    # Statistiques vidéo
    video_stats = ""
    if st.session_state.video_analysis:
        analysis = st.session_state.video_analysis
        total_persons = sum(len(f.get("detections", [])) for f in analysis.get("frames", []))
        allonge = sum(sum(1 for d in f.get("detections", []) if d.get("posture") == "ALLONGE") for f in analysis.get("frames", []))
        debout = sum(sum(1 for d in f.get("detections", []) if d.get("posture") == "DEBOUT") for f in analysis.get("frames", []))
        incertain = sum(sum(1 for d in f.get("detections", []) if d.get("posture") == "INCERTAIN") for f in analysis.get("frames", []))
        
        video_stats = f"""
        <div class="stats">
            <h3>Analyse Vidéo</h3>
            <ul>
                <li>Personnes détectées: <strong>{total_persons}</strong></li>
                <li>Position allongée: <strong>{allonge}</strong></li>
                <li>Position debout: <strong>{debout}</strong></li>
                <li>Position incertaine: <strong>{incertain}</strong></li>
            </ul>
        </div>
        """
    
    # Construire la conversation
    conversation_html = ""
    for msg in st.session_state.conversation_history:
        role_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        content = msg["content"].replace("\n", "<br>")
        conversation_html += f"""
        <div class="message {role_class}">
            <strong>{msg['role'].upper()}:</strong>
            <p>{content}</p>
        </div>
        """
    
    # Status vidéo
    video_status = "Analysée" if st.session_state.video_analysis else "Non analysée"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>Rapport Analyse Drone</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: white;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 900px;
                margin: 0 auto;
            }}
            .header {{
                border-bottom: 3px solid #00d4aa;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .header h1 {{
                color: #00d4aa;
                margin: 0 0 10px 0;
            }}
            .header p {{
                color: #666;
                margin: 0;
            }}
            .stats {{
                background: #f0f9f8;
                border-left: 4px solid #00d4aa;
                padding: 15px;
                margin: 20px 0;
                border-radius: 4px;
            }}
            .stats h3 {{
                margin-top: 0;
                color: #00d4aa;
            }}
            .stats ul {{
                list-style: none;
                padding: 0;
            }}
            .stats li {{
                padding: 5px 0;
            }}
            .conversation {{
                margin-top: 30px;
            }}
            .conversation h2 {{
                color: #00d4aa;
                border-bottom: 2px solid #00d4aa;
                padding-bottom: 10px;
            }}
            .message {{
                margin: 15px 0;
                padding: 12px;
                border-radius: 6px;
                page-break-inside: avoid;
            }}
            .user-msg {{
                background: #e3f2fd;
                border-left: 4px solid #0099ff;
            }}
            .assistant-msg {{
                background: #f0f9f8;
                border-left: 4px solid #00d4aa;
            }}
            .message strong {{
                color: #00d4aa;
            }}
            .message p {{
                margin: 8px 0;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #999;
                font-size: 12px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Rapport d'Analyse Drone Intelligent</h1>
                <p>Généré le {timestamp}</p>
            </div>
            
            <div class="summary">
                <h2 style="color: #00d4aa; border-bottom: 2px solid #00d4aa; padding-bottom: 10px;">Résumé</h2>
                <ul>
                    <li><strong>Messages utilisateur:</strong> {user_msgs}</li>
                    <li><strong>Réponses assistant:</strong> {assistant_msgs}</li>
                    <li><strong>Statut vidéo:</strong> {video_status}</li>
                </ul>
            </div>
            
            {video_stats}
            
            <div class="conversation">
                <h2>Conversation</h2>
                {conversation_html}
            </div>
            
            <div class="footer">
                <p>Ce rapport a été généré automatiquement par Drone Intelligent</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def html_to_pdf(html_string: str) -> bytes:
    """Convertit HTML en PDF."""
    try:
        if HTML is None:
            st.error("WeasyPrint non installé. Installez-le avec: pip install weasyprint")
            return b""
        
        pdf_bytes = BytesIO()
        HTML(string=html_string).write_pdf(pdf_bytes)
        return pdf_bytes.getvalue()
    except Exception as e:
        st.error(f"Erreur lors de la conversion PDF: {str(e)}")
        return b""


# Load video analysis
video_analysis = get_latest_video_analysis()
st.session_state.video_analysis = video_analysis

# Sidebar with settings and history
with st.sidebar:
    st.markdown("### Configuration")
    
    # Refresh models button
    if st.button("Actualiser modèles", use_container_width=True, key="refresh_models_btn"):
        try:
            models = get_available_ollama_models(st.session_state.ollama_base_url)
            if models:
                st.session_state.available_models = models
                st.toast(f"✅ {len(models)} modèle(s) trouvé(s)", icon="✅")
            else:
                st.toast("Aucun modèle trouvé", icon="⚠️")
        except Exception as e:
            st.toast(f"Erreur: {str(e)}", icon="❌")
    
    # Load models on first load
    if not st.session_state.available_models:
        st.session_state.available_models = get_available_ollama_models(st.session_state.ollama_base_url)
    
    st.markdown("---")
    
    # Model selector
    selected = st.selectbox(
        "Modèle LLM",
        st.session_state.available_models,
        index=st.session_state.available_models.index(st.session_state.selected_model) if st.session_state.selected_model in st.session_state.available_models else 0,
        key="model_select"
    )
    if selected != st.session_state.selected_model:
        st.session_state.selected_model = selected
    
    st.markdown("---")
    st.markdown("### Historique")
    
    if st.button("Nouvelle conversation", use_container_width=True):
        new_conversation()
    
    st.markdown("---")
    
    conversations = get_conversation_list()
    if conversations:
        for conv in conversations:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(
                    f"{conv['preview'][:30]}...\n{conv['message_count']} msg",
                    key=f"conv_{conv['id']}",
                    use_container_width=True
                ):
                    load_conversation(conv['id'])
            with col2:
                timestamp = datetime.fromisoformat(conv['timestamp']).strftime("%d/%m")
                st.caption(timestamp)
    else:
        st.caption("Pas d'historique")

# Main chat area
st.title("Drone Intelligent")
st.markdown("Analyseur vidéo avec YOLO + Vision par ordinateur")

# Create a container for all messages - this ensures they stay above the input
message_container = st.container()

# Progress placeholder for video processing
progress_placeholder = st.empty()

# Chat input with upload and report buttons
col1, col2, col3 = st.columns([0.7, 0.15, 0.15])

with col1:
    user_input = st.chat_input("Vos questions...", key="chat_input")

with col2:
    uploaded_file = st.file_uploader("", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed", key="chat_uploader")
    if uploaded_file is not None:
        if st.button("📤", key="process_btn", help="Traiter la vidéo", use_container_width=True):
            # Show progress bar
            with progress_placeholder.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Simulate progress updates
                    for i in range(1, 101, 20):
                        progress_bar.progress(i)
                        status_text.text(f"Traitement YOLO... {i}%")
                    
                    analysis = process_uploaded_video(uploaded_file)
                    
                    if analysis:
                        progress_bar.progress(100)
                        status_text.text("Vidéo traitée!")
                        st.session_state.video_analysis = analysis
                        st.toast("Vidéo traitée!", icon="✅")
                        add_message("assistant", f"✅ J'ai traité la vidéo **{uploaded_file.name}**. Posez-moi vos questions!")
                        
                        # Clear progress after short delay
                        import time
                        time.sleep(1)
                except Exception as e:
                    st.toast(f"Erreur: {str(e)}", icon="❌")
                finally:
                    progress_placeholder.empty()
                    st.rerun()

with col3:
    if st.button("📊", key="report_btn", help="Générer rapport", use_container_width=True):
        if not st.session_state.chat_started:
            st.toast("Commencez une conversation d'abord", icon="⚠️")
        else:
            with st.spinner("Génération rapport..."):
                try:
                    # Générer rapport HTML
                    html_report = generate_html_report()
                    
                    # Convertir en PDF
                    pdf_data = html_to_pdf(html_report)
                    
                    if pdf_data:
                        st.session_state.report_pdf = pdf_data
                        st.session_state.report_generated = True
                        st.toast("Rapport PDF généré!", icon="✅")
                    else:
                        st.toast("Erreur lors de la génération du PDF", icon="❌")
                except Exception as e:
                    st.toast(f"Erreur: {str(e)}", icon="❌")

# Handle user input
if user_input:
    st.session_state.chat_started = True
    add_message("user", user_input)
    
    # Generate response
    with st.spinner("Traitement..."):
        response = chat_with_llm(user_input)
    
    add_message("assistant", response)
    
    # Check if user asked about flying/scanning and show YOLO image + detected faces
    if detect_flying_request(user_input) and st.session_state.video_analysis:
        st.markdown("---")
        
        yolo_img, summary = get_yolo_detection_image()
        if yolo_img is not None:
            st.image(yolo_img, caption=summary, use_column_width=True)
        else:
            st.info(f"ℹ️ {summary}")
        
        # Extract and display detected faces with face comparison
        st.markdown("---")
        extract_and_display_detected_faces()

# Display messages in the message container (always stays above input)
with message_container:
    # Welcome message if no history
    if not st.session_state.conversation_history:
        st.markdown("""
        ### 👋 Bienvenue!
        
        Je suis un assistant d'analyse vidéo de drone. Je peux vous aider à:
        - **Analyser** une zone aérienne pour détecter des personnes et évaluer leur posture et leur risque
        - **Interpréter** les résultats aeriens
        - **Générer** des rapports détaillés d'analyse
        """)
    
    # Display all conversation messages
    for entry in st.session_state.conversation_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

st.markdown("---")

# Download report if available
if st.session_state.get("report_generated") and st.session_state.get("report_pdf"):
    col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
    with col3:
        st.download_button(
            label="⬇️ Rapport PDF",
            data=st.session_state.report_pdf,
            file_name=f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_report"
        )