import os
import base64
from openai import OpenAI, OpenAIError


VLM_BASE_URL = "http://127.0.0.1:11434/v1"
VLM_MODEL = "llava:latest"
VLM_API_KEY = "ollama"


def VLM_Call(image_path: str, text_prompt: str) -> str:
    try:
        if not os.path.isfile(image_path):
            return f"Unexpected Error: image introuvable : {image_path}"

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        client = OpenAI(
            base_url=VLM_BASE_URL,
            api_key=VLM_API_KEY,
            timeout=120,
        )

        response = client.chat.completions.create(
            model=VLM_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": """
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
""",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=512,
        )

        return response.choices[0].message.content or ""

    except OpenAIError as e:
        status_code = getattr(e, "status_code", "N/A")
        return (
            f"Error {status_code}: {str(e)} | "
            f"VLM_BASE_URL={VLM_BASE_URL} | VLM_MODEL={VLM_MODEL}"
        )

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