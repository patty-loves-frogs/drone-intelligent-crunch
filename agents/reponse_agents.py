# agents/response_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


class ResponseAgent(BaseAgent):

    def __init__(self):
        super().__init__("response_agent")
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0
        )

    def run(self, state):
        print("\n🤖 [3/4] Génération de la réponse...")
        mission = state["mission"]
        frames = state["frames"]
        history = state["conversation"]["history"]

        # résumé des détections
        frames_summary = ""
        for event_id, frame in frames.items():
            frames_summary += f"- Event {event_id} :\n"
            frames_summary += f"  posture={frame.get('posture', 'N/A')}\n"
            frames_summary += f"  personnes détectées={frame.get('person_count', 'N/A')}\n"
            frames_summary += f"  description={frame.get('desc', 'N/A')}\n"

        # construire les messages depuis l'historique
        messages = [
            SystemMessage(content=f"""
            Tu es un assistant drone intelligent.
            Tu réponds à l'utilisateur de façon naturelle et concise.
            Tu te bases sur les informations de la mission et les détections.

            === MISSION ===
            Objectif : {mission['goal']}
            Cible : {mission['target_type']}
            Action : {mission['action']}

            === DÉTECTIONS ===
            {frames_summary if frames_summary else "Aucune détection disponible."}
            """)
        ]

        # ajouter l'historique dans les messages
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["message"]))
            elif msg["role"] == "llm":
                messages.append(SystemMessage(content=msg["message"]))

        response = self.llm.invoke(messages)
        answer = response.content

        # ajouter la réponse dans l'historique avec role llm
        state["conversation"]["history"].append({
            "role": "llm",
            "message": answer,
            "timestamp": datetime.now().isoformat()
        })
        print("✅ [3/4] Réponse générée")
        return state


if __name__ == "__main__":

    from core.state.drone_state import state
    import pprint

    state["mission"]["raw_prompt"] = "inspecter le bâtiment pour détecter des personnes au sol"
    state["mission"]["goal"] = "inspection"
    state["mission"]["target_type"] = "personnes au sol"
    state["mission"]["action"] = "inspecter le bâtiment pour détecter des personnes au sol"

    state["frames"] = {
        0: {
            "posture": "ALLONGE",
            "person_count": 2,
            "desc": "Two men are working on a damaged car."
        },
        1: {
            "posture": "DEBOUT",
            "person_count": 1,
            "desc": "A person standing near a vehicle."
        },
    }

    state["conversation"]["history"].append({
        "role": "user",
        "message": state["mission"]["raw_prompt"],
        "timestamp": datetime.now().isoformat()
    })

    agent = ResponseAgent()
    state = agent.run(state)

    pprint.pprint(state, sort_dicts=False)