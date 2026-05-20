from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from yolo import analyze_video_with_yolo, select_vlm_candidates
from vlm_call import VLM_Call
from reporter import reporter_node


class DroneState(TypedDict):
    instruction: str
    video_path: str
    images: List[str]
    detections: List[Dict[str, Any]]
    vlm_candidates: List[Dict[str, Any]]
    vlm_observations: List[Dict[str, Any]]
    report: str
    yolo_json: Dict[str, Any]


# ============================================================
# YOLO NODE
# ============================================================
def yolo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    instruction = state.get("instruction", "")
    video_path = state.get("video_path")

    if not video_path:
        yolo_json = {
            "video_path": "",
            "frames_analyzed": 0,
            "frames": [],
            "detections": [],
            "summary": {
                "persons_detected": 0,
                "standing_count": 0,
                "lying_count": 0,
                "uncertain_count": 0,
                "vlm_candidate_count": 0,
            },
            "error": "Aucune video fournie. Veuillez specifier video_path.",
        }
        return {
            **state,
            "yolo_json": yolo_json,
            "images": [],
            "detections": [],
            "vlm_candidates": [],
        }

    yolo_json = analyze_video_with_yolo(video_path)

    frames = yolo_json.get("frames", [])
    images = [
        frame.get("annotated_image_path") or frame.get("raw_image_path")
        for frame in frames
        if frame.get("annotated_image_path") or frame.get("raw_image_path")
    ]
    detections = yolo_json.get("detections", [])
    vlm_candidates = select_vlm_candidates(yolo_json, instruction)
    yolo_json["summary"]["vlm_candidate_count"] = len(vlm_candidates)

    return {
        **state,
        "yolo_json": yolo_json,
        "images": images,
        "detections": detections,
        "vlm_candidates": vlm_candidates,
    }


# ============================================================
# VLM NODE
# ============================================================
def vlm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    vlm_candidates = state.get("vlm_candidates", [])
    vlm_candidates = vlm_candidates[:3]             #Car compilation trop longue
    instruction = state.get("instruction", "")

    if not vlm_candidates:
        return {**state, "vlm_observations": []}

    observations = []
    for candidate in vlm_candidates:
        image_path = candidate.get("image_path")
        if not image_path:
            continue

        prompt = (
            f"Instruction drone : {instruction}\n\n"
            "Decris ce que tu vois. Y a-t-il des personnes ? "
            "Si oui, combien, ou sont-elles, et quelle est leur posture (debout, allongee, blessee) ?"
        )

        try:
            raw_output = VLM_Call(image_path, prompt)
        except Exception as e:
            observations.append(
                {
                    "image_path": image_path,
                    "error": str(e),
                    "text": "",
                    "person_visible": "N/A",
                    "confidence": "N/A",
                    "description": "",
                    "reason": candidate.get("reason", ""),
                }
            )
            continue

        observations.append(
            {
                "image_path": image_path,
                "error": None,
                "text": raw_output,
                "person_visible": "a determiner",
                "confidence": "N/A",
                "description": raw_output,
                "reason": candidate.get("reason", ""),
            }
        )

    return {**state, "vlm_observations": observations}


# ============================================================
# GRAPHE
# ============================================================
builder = StateGraph(DroneState)
builder.add_node("yolo", yolo_node)
builder.add_node("vlm", vlm_node)
builder.add_node("reporter", reporter_node)

builder.set_entry_point("yolo")
builder.add_edge("yolo", "vlm")
builder.add_edge("vlm", "reporter")
builder.add_edge("reporter", END)

graph = builder.compile()