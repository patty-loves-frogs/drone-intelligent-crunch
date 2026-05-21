import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent
from core.state.drone_state import state
import json
from pathlib import Path
from agents.yolo import run_video
import pprint
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class InspectionAgent(BaseAgent):

    def __init__(self):
        super().__init__("inspection_agent")
        
    def find_analysis_json(self) -> str:
        matches = list(PROJECT_ROOT.rglob("analysis.json"))
        if not matches:
            raise FileNotFoundError("Aucun analysis.json trouvé")
        return str(matches[0])
        
    def extract_first_frame_per_event(self, json_path: str) -> dict:
        with open(json_path, "r") as f:
            data = json.load(f)

        seen_events = set()
        results = {}

        for frame in data["frames"]:
            event_id = frame["event_id"]
            if event_id not in seen_events:
                seen_events.add(event_id)
                results[event_id] = {
                    "raw_image_path": frame["raw_image_path"],
                    "annotated_image_path": frame["annotated_image_path"],
                    "posture": frame["detections"][0]["posture"]
                }

        return results

    def run(self, state):
        print("\n🚁 [1/4] Lancement de l'inspection de la zone...")
        # supprimer le dossier outputs
        outputs_path = PROJECT_ROOT / "outputs"
        if outputs_path.exists():
            shutil.rmtree(outputs_path)
            
        # appeler YOLO
        print("📹 [1/4] Analyse vidéo en cours...")
        camera_path = state["vision"]["camera_source"]
        run_video(camera_path)

        # chercher le json et extraire les frames
        json_path = self.find_analysis_json()
        frames = self.extract_first_frame_per_event(json_path)

        # mettre dans le state
        state["frames"] = frames
        state["mission"]["goal"] = "rapport"
        print("✅ [1/4] Inspection terminée")
        
        return state
   
        
if __name__ == "__main__":   
    agent = InspectionAgent()
    state = agent.run(state)
    pprint.pprint(state)