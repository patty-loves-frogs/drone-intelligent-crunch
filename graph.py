from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from agents.yolo import yolo_node
from agents.vlm import vlm_node
from agents.reporter import reporter_node


class DroneState(TypedDict):
    instruction: str
    images: List[str]
    detections: List[Dict[str, Any]]
    vlm_candidates: List[str]
    vlm_observations: List[Dict[str, Any]]
    report: str


builder = StateGraph(DroneState)
builder.add_node("yolo", yolo_node)
builder.add_node("vlm", vlm_node)
builder.add_node("reporter", reporter_node)

builder.set_entry_point("yolo")
builder.add_edge("yolo", "vlm")
builder.add_edge("vlm", "reporter")
builder.add_edge("reporter", END)

graph = builder.compile()
