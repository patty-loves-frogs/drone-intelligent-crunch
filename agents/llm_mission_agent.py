# agents/llm_mission_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import json
from datetime import datetime
load_dotenv()


class LLMMissionAgent(BaseAgent):

    def __init__(self):
        super().__init__("llm_mission_agent")
        self.llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0
        )

    def run(self, state):

        user_prompt = state["mission"]["raw_prompt"]
        history = state["conversation"]["history"]

        # construire l'historique pour le contexte
        history_text = ""
        for msg in history:
            history_text += f"{msg['role']} : {msg['message']}\n"

        prompt = f"""
        Analyse cette mission drone.

        Tu dois répondre UNIQUEMENT avec un JSON brut.
        Aucun texte avant.
        Aucun texte après.
        Aucun ```json.
        Aucun ``` tout court.
        Aucun markdown.
        Ta réponse doit commencer par {{ et finir par }}.

        === HISTORIQUE DE LA CONVERSATION ===
        {history_text if history_text else "Aucun historique."}

        === NOUVEAU MESSAGE ===
        {user_prompt}

        Attention : les missions ne peuvent être que : inspection, reponse, rapport. Ces valeurs partent dans le champ goal du json.
        Si l'historique montre que l'utilisateur veut continuer ou affiner une mission existante, tiens en compte pour définir le goal.

        Format attendu :

        {{
            "goal": "...",
            "target_type": "...",
            "priority": "...",
            "action": "..."
        }}
        """

        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])

        mission_data = json.loads(response.content)

        state["mission"]["goal"] = mission_data["goal"]
        state["mission"]["target_type"] = mission_data["target_type"]
        state["mission"]["action"] = mission_data["action"]

        state["conversation"]["history"].append({
            "role": "user",
            "message": user_prompt,
            "timestamp": datetime.now().isoformat()
        })

        return state


if __name__ == "__main__":

    from core.state.drone_state import state
    import pprint

    # simuler un premier tour
    state["mission"]["raw_prompt"] = "inspecter le bâtiment pour détecter des personnes au sol"
    agent = LLMMissionAgent()
    state = agent.run(state)

    # simuler une réponse du llm dans l'historique
    state["conversation"]["history"].append({
        "role": "llm",
        "message": "J'ai détecté deux personnes allongées et une debout. Souhaitez-vous affiner ?",
        "timestamp": datetime.now().isoformat()
    })

    # simuler un deuxième tour
    state["mission"]["raw_prompt"] = "oui concentre toi sur la zone gauche"
    state = agent.run(state)

    pprint.pprint(state, sort_dicts=False)