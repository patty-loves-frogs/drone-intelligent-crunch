from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # remonte jusqu'à la racine

state = {
    "mission": {
        "raw_prompt": "",
        "goal": "",
        "target_type": "",
        "action": ""
    },
    "vision": {
        "camera_source": str(PROJECT_ROOT / "video" / "vid1.MP4")
    },
    "frames": {},
    "conversation": {
        "history": []
    }
}