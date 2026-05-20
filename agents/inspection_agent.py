import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from core.state.drone_state import state
import json
from pathlib import Path
from yolo import run_video
import pprint
import shutil


class InspectionAgent(BaseAgent):

    def __init__(self):
        super().__init__("inspection_agent")
        
    def find_analysis_json(self, project_root: str) -> str:
        matches = list(Path(project_root).rglob("analysis.json"))
        if not matches:
            raise FileNotFoundError("Aucun analysis.json trouvé")
        return str(matches[0])
        
    def extract_first_frame_per_event(self, json_path: str) -> dict:
        with open(json_path, "r") as f:
            data = json.load(f)

        seen_events = set()
        results = {}  # dict au lieu de list

        for frame in data["frames"]:
            event_id = frame["event_id"]
            if event_id not in seen_events:
                seen_events.add(event_id)
                results[event_id] = {  # event_id comme clé
                    "raw_image_path": frame["raw_image_path"],
                    "annotated_image_path": frame["annotated_image_path"]
                }

        return results

    def run(self, state):

        
        #supprimer le dossier outputs
        outputs_path = Path("/Users/User/drone-intelligent-crunch/outputs")
        if outputs_path.exists():
            shutil.rmtree(outputs_path)
            
        # appeler YOLO
        camera_path = state["vision"]["camera_source"]
        run_video(camera_path)

        # chercher le json et extraire les frames
        json_path = self.find_analysis_json("/Users/User/drone-intelligent-crunch")
        frames = self.extract_first_frame_per_event(json_path)

        
        # mettre dans le state
        state["frames"] = frames
        state["mission"]["goal"] = "rapport"
        
        return state
   
   
        
if __name__ == "__main__":   
    agent = InspectionAgent()
    state = agent.run(state)
    pprint.pprint(state)

    