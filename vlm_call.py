import os
import base64
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

def VLM_Call(image_path: str, text_prompt: str) -> str:
    try:
        VLM_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        VLM_MODEL = os.getenv("OLLAMA_MODEL", "llava:latest")
        if not os.path.isfile(image_path):
            return f"Unexpected Error: image introuvable : {image_path}"

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        llm = ChatOllama(
            model=VLM_MODEL,
            base_url=VLM_BASE_URL,
            temperature=0.0
        )
        
        # Le format attendu
        system_content = """
Output ONLY valid JSON.

JSON Schema:
{
  "desc": "1 sentence image summary",
  "person_count": 0,
  "locs": ["Person 1 location", "Person 2 location"],
  "non_std": "yes/no"
}

Constraints:
- person_count: total count, exclude shadows/reflections.
- locs: be extremely concise, e.g. "back-left", "center-fore".
- non_std: "no" only if all people are standing and healthy.
- non_std: "yes" if anyone is crouching, lying, under debris, injured, or in an unusual posture.
"""
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
        return (
            f"Unexpected Error: {str(e)} | "
            f"VLM_BASE_URL={VLM_BASE_URL} | VLM_MODEL={VLM_MODEL}"
        )


if __name__ == "__main__":
    test_image = os.getenv("VLM_TEST_IMAGE", "")
    test_prompt = "Décris l'image et indique s'il y a une personne au sol."

    if not test_image:
        print("Définis VLM_TEST_IMAGE pour tester, exemple :")
        print("set VLM_TEST_IMAGE=outputs/runs/.../raw/event_000_frame_000440_0.jpg")
    else:
        print(VLM_Call(test_image, test_prompt))