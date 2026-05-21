# agents/report_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ReportAgent(BaseAgent):

    def __init__(self):
        super().__init__("report_agent")

    def run(self, state):
        print("\n📄 [4/4] Génération du rapport PDF...")
        mission = state["mission"]
        frames = state["frames"]
        history = state["conversation"]["history"]

        # chemin de sortie du PDF
        output_path = output_path = str(PROJECT_ROOT / "reports" / "rapport.pdf")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('title', parent=styles['Title'], fontSize=20, textColor=colors.HexColor('#1a1a2e'))
        heading_style = ParagraphStyle('heading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#16213e'))
        normal_style = styles['Normal']
        label_style = ParagraphStyle('label', parent=styles['Normal'], textColor=colors.grey, fontSize=9)

        story = []

        # titre
        story.append(Paragraph("Rapport de Mission Drone", title_style))
        story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}", label_style))
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e')))
        story.append(Spacer(1, 0.5*cm))

        # infos mission
        story.append(Paragraph("Mission", heading_style))
        story.append(Paragraph(f"<b>Objectif :</b> {mission['goal']}", normal_style))
        story.append(Paragraph(f"<b>Cible :</b> {mission['target_type']}", normal_style))
        story.append(Paragraph(f"<b>Action :</b> {mission['action']}", normal_style))
        story.append(Paragraph(f"<b>Prompt :</b> {mission['raw_prompt']}", normal_style))
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 0.5*cm))

        # réponse llm
        last_response = next(
            (msg["message"] for msg in reversed(history) if msg["role"] == "llm"),
            "Aucune réponse"
        )
        story.append(Paragraph("Synthèse", heading_style))
        story.append(Paragraph(last_response, normal_style))
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
        story.append(Spacer(1, 0.5*cm))

        # détections par event
        story.append(Paragraph("Détections par Event", heading_style))
        story.append(Spacer(1, 0.3*cm))

        for event_id, frame in frames.items():
            story.append(Paragraph(f"Event {event_id}", ParagraphStyle('event', parent=styles['Heading3'], textColor=colors.HexColor('#e94560'))))
            story.append(Paragraph(f"<b>Posture :</b> {frame.get('posture', 'N/A')}", normal_style))
            story.append(Paragraph(f"<b>Personnes détectées :</b> {frame.get('person_count', 'N/A')}", normal_style))
            story.append(Paragraph(f"<b>Description :</b> {frame.get('desc', 'N/A')}", normal_style))
            story.append(Spacer(1, 0.3*cm))

            # image annotée
            img_path = frame.get("annotated_image_path")
            if img_path and Path(img_path).exists():
                img = Image(img_path, width=12*cm, height=7*cm)
                story.append(img)
            else:
                story.append(Paragraph("<i>Image non disponible</i>", label_style))

            story.append(Spacer(1, 0.5*cm))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            story.append(Spacer(1, 0.3*cm))

        doc.build(story)

        state["report_path"] = output_path
        print(f"[ReportAgent] PDF généré : {output_path}")
        print(f"✅ [4/4] Rapport sauvegardé : {output_path}")

        return state


if __name__ == "__main__":
    from core.state.drone_state import state
    import pprint

    state["mission"]["raw_prompt"] = "Inspecte la zone et dis moi ce que tu vois"
    state["mission"]["goal"] = "inspection"
    state["mission"]["target_type"] = "zone"
    state["mission"]["action"] = "Inspecte la zone et rapporte ce qui est visible"

    state["frames"] = {
        0: {
            "annotated_image_path": r"C:\Users\User\drone-intelligent-crunch\outputs\runs\20260521_111003_vid1\annotated\event_000_frame_000420_0.jpg",
            "posture": "DEBOUT",
            "desc": "A view of a concrete lot with overgrown vegetation.",
            "person_count": 0
        },
        1: {
            "annotated_image_path": r"C:\Users\User\drone-intelligent-crunch\outputs\runs\20260521_111003_vid1\annotated\event_001_frame_000805_0.jpg",
            "posture": "DEBOUT",
            "desc": "Two men are working on a severely damaged grey car.",
            "person_count": 2
        }
    }

    state["conversation"]["history"].append({
        "role": "llm",
        "message": "J'observe deux personnes debout dans la zone inspectée.",
        "timestamp": datetime.now().isoformat()
    })

    agent = ReportAgent()
    state = agent.run(state)

    pprint.pprint(state, sort_dicts=False)