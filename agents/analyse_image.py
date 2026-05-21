# agents/vlm_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from agents.base_agent import BaseAgent
from agents.vlm_call import VLM_Call


class VLMAgent(BaseAgent):

    def __init__(self):
        super().__init__("vlm_agent")

    def run(self, state):

        frames = state["frames"]

        for event_id, frame in frames.items():
            image_path = frame["raw_image_path"]
            result = json.loads(VLM_Call(image_path, "Analyse this image."))

            frame["desc"] = result.get("desc", "")
            frame["person_count"] = result.get("person_count", 0)

        state["frames"] = frames

        print("[VLMAgent] Analyse VLM terminée")

        return state


if __name__ == "__main__":
    from core.state.drone_state import state
    import pprint

    state["frames"] = {
        0: {
            "raw_image_path": r"C:\Users\User\drone-intelligent-crunch\outputs\runs\20260521_093705_vid1\raw\event_001_frame_000795_1.jpg",
            "annotated_image_path": r"C:\Users\User\drone-intelligent-crunch\outputs\runs\20260521_093705_vid1\raw\event_001_frame_000795_1.jpg",
            "posture": "DEBOUT"
        }
    }

    agent = VLMAgent()
    state = agent.run(state)

    pprint.pprint(state["frames"], sort_dicts=False)