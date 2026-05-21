# core/runtime/runtime.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.orchestrator.graph import graph


class DroneRuntime:

    def __init__(self):
        self.graph = graph

    def run(self, user_prompt, state):

        # injecter la mission utilisateur
        state["mission"]["raw_prompt"] = user_prompt

        # lancer le graph
        result_state = self.graph.invoke(state)

        return result_state