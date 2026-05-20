import os
import json
from datetime import datetime
from typing import Tuple, List, Dict, Any
import gradio as gr
from graph import graph
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class ConversationManager:
    """Gère la conversation et l'historique."""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_analysis: Dict[str, Any] = {}
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.initial_instruction: str = ""
    
    def set_initial_instruction(self, instruction: str):
        """Définit l'instruction initiale pour l'analyse."""
        self.initial_instruction = instruction
        self.add_user_message(instruction)
        self.add_system_response(f"Mission enregistrée: _{instruction}_\n\n💬 Posez vos questions ou précisions...")
    
    def add_user_message(self, message: str):
        """Ajoute un message utilisateur."""
        self.conversation_history.append({
            "role": "user",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_system_response(self, message: str):
        """Ajoute une réponse système."""
        self.conversation_history.append({
            "role": "system",
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def set_analysis_result(self, analysis: Dict[str, Any]):
        """Stocke le résultat de l'analyse."""
        self.current_analysis = analysis
    
    def get_chat_display(self) -> str:
        """Retourne le chat formaté pour l'affichage."""
        if not self.conversation_history:
            return "Aucune conversation pour le moment. Définissez d'abord une mission !"
        
        display = ""
        for entry in self.conversation_history:
            role = "👤 Vous" if entry["role"] == "user" else "🤖 Système"
            timestamp = entry["timestamp"].split("T")[1][:5]  # HH:MM
            display += f"\n**{role}** ({timestamp})\n"
            display += f"{entry['message']}\n"
        
        return display
    
    def get_report(self) -> str:
        """Génère un rapport récapitulatif de la conversation."""
        if not self.conversation_history:
            return """### Rapport de Conversation

Aucune conversation pour le moment.

**Étapes**:
1. Définissez une mission
2. Discutez avec le chatbot
3. Le rapport s'affichera ici
"""
        
        report = f"""# Rapport de Conversation - {self.session_id}

## Mission
**{self.initial_instruction}**

---

## Discussion

"""
        
        for entry in self.conversation_history:
            role = " **Vous**" if entry["role"] == "user" else " **Chatbot**"
            report += f"\n{role}\n"
            report += f"{entry['message']}\n\n"
        
        return report
    
    def reset(self):
        """Réinitialise la conversation."""
        self.conversation_history = []
        self.current_analysis = {}
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.initial_instruction = ""


# Instance globale du gestionnaire
manager = ConversationManager()


def check_images_available() -> bool:
    """Vérifie si des images sont disponibles dans frames/."""
    frames_dir = "frames"
    if not os.path.isdir(frames_dir):
        return False
    return any(
        f.lower().endswith((".jpg", ".jpeg", ".png"))
        for f in os.listdir(frames_dir)
    )


def set_mission(mission_text: str) -> Tuple[str, str]:
    """Définit la mission et initialise la conversation."""
    if not mission_text.strip():
        return manager.get_chat_display(), manager.get_report()
    
    if not check_images_available():
        chat = " Vous\n" + mission_text + "\n\n Système\n❌ Aucune image trouvée dans le dossier `frames/`. Veuillez ajouter des images."
        return chat, manager.get_report()
    
    manager.set_initial_instruction(mission_text)
    return manager.get_chat_display(), manager.get_report()


def chat_with_system(user_message: str) -> Tuple[str, str]:
    """Ajoute un message utilisateur au chat et génère une réponse via Ollama."""
    if not user_message.strip():
        return manager.get_chat_display(), manager.get_report()
    
    if not manager.initial_instruction:
        error_msg = " Veuillez d'abord définir une mission !"
        return manager.get_chat_display(), manager.get_report()
    
    # Ajouter le message utilisateur
    manager.add_user_message(user_message)
    
    # Construire le contexte pour le LLM
    analysis = manager.current_analysis
    
    # Résumé des détections YOLO
    yolo_summary = "Aucune détection YOLO effectuée."
    if analysis and "detections" in analysis:
        yolo_dets = [d for d in analysis["detections"] if d.get("label") != "error"]
        if yolo_dets:
            yolo_lines = [f"- {d['label']} sur {os.path.basename(d['image'])} (conf: {d['confidence']})" for d in yolo_dets]
            yolo_summary = "\n".join(yolo_lines)
    
    # Résumé des observations VLM
    vlm_summary = "Aucune analyse VLM effectuée."
    if analysis and "vlm_observations" in analysis:
        vlm_obs = analysis["vlm_observations"]
        if vlm_obs:
            vlm_lines = [f"- {os.path.basename(obs['image'])}: personne={obs['person_visible']}, confiance={obs['confidence']}" for obs in vlm_obs]
            vlm_summary = "\n".join(vlm_lines)
    
    # Créer le prompt pour le LLM
    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "Tu es un assistant de surveillance par drone. Tu aides l'utilisateur à analyser et discuter des résultats d'analyse d'images. "
            "Réponds toujours en français et sois précis et utile."
        ),
        (
            "human",
            (
                "Mission : {instruction}\n\n"
                "Résultats actuels d'analyse :\n"
                "Détections YOLO :\n{yolo_summary}\n\n"
                "Observations VLM :\n{vlm_summary}\n\n"
                "Question de l'utilisateur : {user_message}\n\n"
                "Réponds à la question de l'utilisateur en te basant sur les résultats d'analyse disponibles. "
                "Si une information n'est pas disponible, indique-le poliment."
            ),
        ),
    ])
    
    try:
        # Appeler Ollama pour générer une réponse
        llm = ChatOllama(model="qwen2.5:3b", temperature=0.3)
        chain = prompt_template | llm
        
        response_obj = chain.invoke({
            "instruction": manager.initial_instruction,
            "yolo_summary": yolo_summary,
            "vlm_summary": vlm_summary,
            "user_message": user_message,
        })
        
        response = response_obj.content.strip()
    except Exception as e:
        response = f" Erreur lors de la génération de la réponse: {str(e)}\n\n*Assurez-vous que Ollama est en cours d'exécution avec le modèle 'qwen2.5:3b'*"
    
    manager.add_system_response(response)
    
    return manager.get_chat_display(), manager.get_report()


def generate_report() -> Tuple[str, str]:
    """Lance l'analyse et génère le rapport final."""
    
    if not manager.initial_instruction:
        error = " Aucune mission définie. Définissez une mission d'abord !"
        manager.add_system_response(error)
        return manager.get_chat_display(), manager.get_report()
    
    if not check_images_available():
        error = " Aucune image trouvée dans le dossier `frames/`."
        manager.add_system_response(error)
        return manager.get_chat_display(), manager.get_report()
    
    manager.add_system_response(" Analyse en cours... (YOLO → VLM → Rapport)")
    
    try:
        # Exécuter l'analyse via le graphe avec l'instruction initiale
        initial_state = {
            "instruction": manager.initial_instruction,
            "images": [],
            "detections": [],
            "vlm_candidates": [],
            "vlm_observations": [],
            "report": "",
        }
        
        final_state = graph.invoke(initial_state)
        manager.set_analysis_result(final_state)
        
        # Construire la réponse finale
        system_response = "✅ Analyse complète ! Rapport généré ci-dessous."
        manager.add_system_response(system_response)
        
    except Exception as e:
        error_msg = f" Erreur lors de l'analyse: {str(e)}"
        manager.add_system_response(error_msg)
    
    return manager.get_chat_display(), manager.get_report()


def reset_conversation():
    """Réinitialise la conversation."""
    manager.reset()
    return manager.get_chat_display(), manager.get_report()


# Interface Gradio
def create_interface():
    with gr.Blocks(title="Drone Intelligent - Dashboard", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
#  Drone Intelligent Conversationnel
*Analyser vos images de drone via conversation naturelle*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Images")
                images_status = gr.Markdown(" Images détectées")
                
                def update_status():
                    status = " Images trouvées" if check_images_available() else " Aucune image (ajouter dans `frames/`)"
                    return status
                
                images_status.value = update_status()
            
            with gr.Column(scale=1):
                gr.Markdown("###  Modèles")
                gr.Markdown("""
- **YOLO**: yolov8n (détection)
- **VLM**: llava (vision)
- **LLM**: qwen2.5:3b (rapport)
                """)
        
        gr.Markdown("---")
        
        # Étape 1: Définir la mission
        gr.Markdown("##  Étape 1: Définir la Mission")
        with gr.Row():
            mission_input = gr.Textbox(
                label="Mission / Instruction",
                placeholder="Ex: Détecte les personnes au sol dans les images. Identifie leur position.",
                lines=3
            )
        with gr.Row():
            define_mission_btn = gr.Button(" Définir la Mission", scale=1, variant="primary")
        
        gr.Markdown("---")
        
        # Étape 2: Chat avec le système
        gr.Markdown("##  Étape 2: Discuter avec le Chatbot")
        with gr.Row():
            with gr.Column(scale=1):
                chat_display = gr.Markdown("""### Chat
                
Définissez d'abord une mission pour commencer !
                """)
                
                chat_input = gr.Textbox(
                    label="Votre message",
                    placeholder="Ex: Peux-tu aussi vérifier les véhicules ?",
                    lines=2
                )
                
                with gr.Row():
                    send_chat_btn = gr.Button(" Envoyer", scale=2)
                    reset_btn = gr.Button("🔄 Réinitialiser", scale=1)
        
        gr.Markdown("---")
        
        # Étape 3: Générer le rapport
        gr.Markdown("##  Étape 3: Générer le Rapport")
        
        with gr.Row():
            with gr.Column(scale=1):
                generate_btn = gr.Button(" Générer Rapport", scale=1, variant="primary", size="lg")
            with gr.Column(scale=2):
                gr.Markdown("Cliquez ici quand vous êtes prêt pour lancer l'analyse et générer le rapport final.")
        
        gr.Markdown("---")
        
        # Affichage du rapport
        gr.Markdown("##  Rapport Final")
        with gr.Row():
            report_display = gr.Markdown("""
###  Rapport d'Analyse

Aucune analyse effectuée pour le moment.

**Étapes**:
1. Définissez une mission
2. Discutez avec le chatbot
3. Cliquez sur "Générer Rapport" pour analyser
            """)
        
        # Événements
        define_mission_btn.click(
            fn=set_mission,
            inputs=mission_input,
            outputs=[chat_display, report_display]
        )
        
        send_chat_btn.click(
            fn=chat_with_system,
            inputs=chat_input,
            outputs=[chat_display, report_display]
        ).then(
            lambda: "",
            outputs=chat_input
        )
        
        generate_btn.click(
            fn=generate_report,
            outputs=[chat_display, report_display]
        )
        
        reset_btn.click(
            fn=reset_conversation,
            outputs=[chat_display, report_display]
        )
    
    return demo


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
