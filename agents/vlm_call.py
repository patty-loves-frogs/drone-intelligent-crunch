import os
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def VLM_Call(image_path, text_prompt):
    try:
        VLM_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        VLM_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        llm = ChatOllama(
            model=VLM_MODEL,
            base_url=VLM_BASE_URL,
            temperature=0.0
        )
        
        system_content = """Output ONLY valid JSON.
                                    JSON Schema:
                                    "{
                                    "desc": "1 sentence image summary",
                                    "person_count": 0,
                                    "locs": ["Person 1 location", "Person 2 location"],
                                    "non_std": "yes/no"
                                    }"

                                    Constraints:
                                    person_count: Total count (exclude shadows/reflections).
                                    locs: Be extremely concise (e.g., "Back-left", "Center-fore").
                                    non_std: "no" only if all are standing and healthy. "yes" if any are crouching, lying, under debris, or injured."""
                                    
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(
                content=[
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ]
            ),
        ]
        
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

path = r"C:\Users\chouh\Pictures\Screenshots\Screenshot 2026-03-07 210231.png"
prompt = ""

result = VLM_Call(path, prompt)
print(result)
