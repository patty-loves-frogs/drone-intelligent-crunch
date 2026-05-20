# agents/llm_mission_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
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

        prompt = f"""
        Analyse cette mission drone.

        Tu dois répondre UNIQUEMENT avec un JSON brut.
        Aucun texte avant.
        Aucun texte après.
        Aucun ```json.
        Aucun ``` tout court.
        Aucun markdown.
        Ta réponse doit commencer par {{ et finir par }}.

        Mission :
        {user_prompt}

        Attention : les missions ne peuvent être que : inspection, reportage, question.Attention ces valeurs partent dans le champ goal du json

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

    # simuler un prompt utilisateur
    state["mission"]["raw_prompt"] = "inspecter le bâtiment pour détecter des personnes au sol"

    agent = LLMMissionAgent()
    state = agent.run(state)

    pprint.pprint(state, sort_dicts=False)