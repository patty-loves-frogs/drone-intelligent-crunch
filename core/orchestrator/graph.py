from langgraph.graph import StateGraph, END
from agents.llm_mission_agent import LLMMissionAgent
from agents.inspection_agent import InspectionAgent
from agents.reponse_agents import ResponseAgent
from agents.analyse_image import VLMAgent
from agents.report_agent import ReportAgent


# agents
mission_agent = LLMMissionAgent()
vision_agent = InspectionAgent()
vlm_agent = VLMAgent()
reponse_agent = ResponseAgent()
report_agent = ReportAgent()


builder = StateGraph(dict)

# nodes
builder.add_node("mission", mission_agent.run)
builder.add_node("vision", vision_agent.run)
builder.add_node("vlm", vlm_agent.run)
builder.add_node("reponse", reponse_agent.run)
builder.add_node("report", report_agent.run)

builder.set_entry_point("mission")

builder.add_edge("mission", "vision")
builder.add_edge("vision", "vlm")
builder.add_edge("vlm", "reponse")
builder.add_edge("reponse", "report")
builder.add_edge("report", END)

graph = builder.compile()