# app.py
import streamlit as st
import copy
from core.runtime.runtime import DroneRuntime
from core.state.drone_state import state

st.set_page_config(page_title="Drone Intelligence", page_icon="🚁", layout="wide")

st.title("🚁 Drone Intelligence System")

# initialiser le runtime et le state dans la session
if "runtime" not in st.session_state:
    st.session_state.runtime = DroneRuntime()

if "state" not in st.session_state:
    st.session_state.state = copy.deepcopy(state)

if "history" not in st.session_state:
    st.session_state.history = []

# afficher l'historique des messages
for msg in st.session_state.history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["message"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["message"])
            if "pdf" in msg:
                with open(msg["pdf"], "rb") as f:
                    st.download_button(
                        label="📄 Télécharger le rapport PDF",
                        data=f,
                        file_name="rapport.pdf",
                        mime="application/pdf",
                        key=f"dl_{msg['message'][:10]}"
                    )

# input utilisateur
user_prompt = st.chat_input("Mission drone...")

if user_prompt:

    # afficher le message user
    with st.chat_message("user"):
        st.write(user_prompt)

    st.session_state.history.append({"role": "user", "message": user_prompt})

    # lancer le workflow
    with st.chat_message("assistant"):
        with st.status("Traitement en cours...", expanded=True) as status:
            st.write("🚁 Lancement de l'inspection de la zone...")
            st.write("📹 Analyse vidéo en cours...")
            result_state = st.session_state.runtime.run(user_prompt, st.session_state.state)
            st.session_state.state = result_state
            st.write("🔍 Description des images terminée...")
            st.write("🤖 Génération de la réponse...")
            st.write("📄 Génération du rapport PDF...")
            status.update(label="✅ Traitement terminé", state="complete")

        # récupérer la dernière réponse llm
        history = result_state["conversation"]["history"]
        last_response = next(
            (msg["message"] for msg in reversed(history) if msg["role"] == "llm"),
            "Pas de réponse"
        )

        st.write(last_response)

        # proposer le PDF si disponible
        pdf_path = result_state.get("report_path")
        msg_entry = {"role": "assistant", "message": last_response}

        if pdf_path:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Télécharger le rapport PDF",
                    data=f,
                    file_name="rapport.pdf",
                    mime="application/pdf",
                    key="dl_current"
                )
            msg_entry["pdf"] = pdf_path

        st.session_state.history.append(msg_entry)