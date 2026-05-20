import os
import pandas as pd
import streamlit as st

from graph import yolo_node, vlm_node
from reporter import reporter_node


st.set_page_config(page_title="Drone Intelligent Conversationnel", layout="wide")

st.title("Drone Intelligent Conversationnel")
st.markdown("MVP basé sur LangGraph, YOLOv8, VLM local et LLM Reporter.")

video_path = st.text_input(
    "Chemin de la vidéo à analyser",
    placeholder="Ex: videos/mission.mp4",
)

instruction = st.text_input(
    "Instruction / Mission",
    placeholder="Ex: Dis-moi s'il y a quelqu'un au sol dans les images.",
)

if st.button("Analyser"):
    if not instruction:
        st.warning("Veuillez entrer une instruction.")

    elif not video_path or not os.path.isfile(video_path):
        st.error(f"Vidéo introuvable : {video_path}")

    else:
        progress = st.progress(0)
        status = st.empty()

        initial_state = {
            "instruction": instruction,
            "video_path": video_path,
            "images": [],
            "detections": [],
            "vlm_candidates": [],
            "vlm_observations": [],
            "report": "",
            "yolo_json": {},
        }

        status.info("Étape 1/3 : analyse vidéo avec YOLO...")
        progress.progress(15)

        state = yolo_node(initial_state)

        status.success("YOLO terminé.")
        progress.progress(50)

        status.info("Étape 2/3 : analyse VLM des images critiques...")
        progress.progress(60)

        state = vlm_node(state)

        status.success("VLM terminé.")
        progress.progress(80)

        status.info("Étape 3/3 : génération du rapport de mission...")
        progress.progress(90)

        final_state = reporter_node(state)

        status.success("Analyse terminée.")
        progress.progress(100)

        yolo_json = final_state.get("yolo_json", {})
        summary = yolo_json.get("summary", {})

        persons = summary.get("persons_detected", 0)
        events = summary.get("events_count", 0)
        lying = summary.get("lying_count", 0)
        uncertain = summary.get("uncertain_count", 0)
        vlm_count = summary.get("vlm_candidate_count", 0)

        if lying > 0:
            risk = "ÉLEVÉ"
            decision = "Vérification humaine prioritaire"
        elif uncertain > 0 or vlm_count > 0:
            risk = "À VÉRIFIER"
            decision = "Contrôle humain recommandé"
        else:
            risk = "FAIBLE"
            decision = "Aucune alerte critique détectée"

        st.divider()
        st.header("Dashboard mission drone")

        st.subheader("Décision opérationnelle")

        if risk == "ÉLEVÉ":
            st.error(f"Risque global : {risk} — {decision}")
        elif risk == "À VÉRIFIER":
            st.warning(f"Risque global : {risk} — {decision}")
        else:
            st.success(f"Risque global : {risk} — {decision}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Personnes détectées", persons)
        col2.metric("Événements", events)
        col3.metric("Allongées", lying)
        col4.metric("Incertaines", uncertain)
        col5.metric("Images VLM", vlm_count)

        st.subheader("Preuves visuelles annotées")

        images = final_state.get("images", [])

        if images:
            cols = st.columns(3)

            for idx, img_path in enumerate(images):
                with cols[idx % 3]:
                    if os.path.isfile(img_path):
                        st.image(
                            img_path,
                            caption=os.path.basename(img_path),
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Image introuvable : {img_path}")
        else:
            st.info("Aucune image annotée disponible.")

        st.subheader("Timeline des événements détectés")

        rows = []

        for frame in yolo_json.get("frames", []):
            for det in frame.get("detections", []):
                rows.append(
                    {
                        "Événement": frame.get("event_id"),
                        "Frame": frame.get("frame_index"),
                        "Temps (s)": frame.get("timestamp_sec"),
                        "Posture": det.get("posture"),
                        "Risque": det.get("risk_level"),
                        "Confiance YOLO": round(det.get("yolo_confidence", 0), 2),
                        "Incertitude": det.get("uncertainty_score"),
                        "Position": det.get("position_in_image"),
                        "Image annotée": frame.get("annotated_image_path"),
                    }
                )

        if rows:
            events_df = pd.DataFrame(rows)
            st.dataframe(events_df, use_container_width=True)
        else:
            st.info("Aucun événement détecté.")

        st.subheader("Analyses VLM")

        vlm_observations = final_state.get("vlm_observations", [])

        if vlm_observations:
            for obs in vlm_observations:
                image_name = os.path.basename(obs.get("image_path", "image inconnue"))

                with st.expander(f"Analyse VLM — {image_name}"):
                    st.write(f"**Texte :** {obs.get('text', '')}")
                    st.write(f"**Description :** {obs.get('description', '')}")
                    st.write(f"**Raison :** {obs.get('reason', '')}")
        else:
            st.info("Le VLM n'a pas été appelé ou aucune image critique n'a été retenue.")

        st.subheader("Rapport de mission")

        report_text = final_state.get("report", "")
        st.markdown(report_text)

        run_dir = yolo_json.get("output_paths", {}).get("run_dir", "outputs")
        os.makedirs(run_dir, exist_ok=True)

        md_path = os.path.join(run_dir, "rapport_mission.md")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        with open(md_path, "rb") as f:
            st.download_button(
                label="Exporter le rapport de mission (.md)",
                data=f,
                file_name="rapport_mission.md",
                mime="text/markdown",
            )