import os
import streamlit as st
from graph import graph

st.set_page_config(page_title="Drone Intelligent Conversationnel", layout="wide")
st.title("Drone Intelligent Conversationnel")
st.markdown("MVP basé sur LangGraph, YOLOv8, VLM (Ollama) et LLM Reporter.")

instruction = st.text_input(
    "Instruction / Mission",
    placeholder="Ex: Dis-moi s'il y a quelqu'un au sol dans les images."
)

if st.button("Analyser"):
    if not instruction:
        st.warning("Veuillez entrer une instruction.")
    else:
        frames_dir = "frames"
        has_images = (
            os.path.isdir(frames_dir)
            and any(
                f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(frames_dir)
            )
        )
        if not has_images:
            st.error("Aucune image trouvée dans le dossier frames/")
        else:
            with st.spinner("Analyse en cours... YOLO → VLM fallback → Rapport"):
                initial_state = {
                    "instruction": instruction,
                    "images": [],
                    "detections": [],
                    "vlm_candidates": [],
                    "vlm_observations": [],
                    "report": "",
                }
                final_state = graph.invoke(initial_state)

            # --- Affichage des images ---
            st.subheader("Images analysées")
            if final_state["images"]:
                cols = st.columns(min(len(final_state["images"]), 4))
                for idx, img_path in enumerate(final_state["images"]):
                    with cols[idx % len(cols)]:
                        st.image(
                            img_path,
                            caption=os.path.basename(img_path),
                            use_container_width=True,
                        )
            else:
                st.info("Aucune image chargée.")

            # --- Affichage des détections YOLO ---
            st.subheader("Détections YOLO")
            yolo_dets = [d for d in final_state["detections"] if d.get("label") != "error"]
            yolo_errors = [d for d in final_state["detections"] if d.get("label") == "error"]
            if yolo_dets:
                for det in yolo_dets:
                    st.write(
                        f"🎯 `{det['label']}` sur `{os.path.basename(det['image'])}` — "
                        f"conf: {det['confidence']} — bbox: {det['bounding_box']}"
                    )
            else:
                st.info("Aucune détection YOLO pour les classes cibles.")
            if yolo_errors:
                for err in yolo_errors:
                    st.warning(
                        f"⚠️ Erreur YOLO sur `{os.path.basename(err['image'])}` : {err.get('error')}"
                    )

            # --- Affichage VLM ---
            st.subheader("Analyses VLM (fallback)")
            if final_state["vlm_observations"]:
                for obs in final_state["vlm_observations"]:
                    with st.expander(f"VLM : {os.path.basename(obs['image'])}"):
                        st.write(f"**Personne visible :** {obs['person_visible']}")
                        st.write(f"**Confiance :** {obs['confidence']}")
                        st.write(f"**Description :** {obs['description']}")
                        st.write(f"**Raison :** {obs['reason']}")
            else:
                st.info(
                    "Le VLM n'a pas été appelé (détections YOLO suffisantes ou "
                    "instruction ne concernant pas de personne)."
                )

            # --- Rapport final ---
            st.subheader("Rapport Final")
            st.markdown(final_state["report"])
