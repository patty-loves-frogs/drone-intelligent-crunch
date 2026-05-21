import base64
from openai import OpenAI, OpenAIError

def VLM_Call(image_path, text_prompt):
    print("\n🔍 [2/4] Description des images en cours...")
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-no-key-required")
        response = client.chat.completions.create(
            temperature=0.0,
            reasoning_effort=0,
            model="qwen3.5",
            messages=[
                {
                    "role": "system",
                    "content" : """Output ONLY valid JSON.
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
                                    non_std: "no" only if all are standing and healthy. "yes" if any are crouching, lying, under debris, or injured.Reply in french only"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=-1
        )
        print("✅ [2/4] Description des images terminée")
        return response.choices[0].message.content
    except OpenAIError as e:
        status_code = getattr(e, 'status_code', 'N/A')
        return f"Error {status_code}: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


if __name__ == "__main__":
    path = r"C:\Users\User\drone-intelligent-crunch\outputs\runs\20260521_093705_vid1\raw\event_001_frame_000795_1.jpg"
    prompt = ""

    result = VLM_Call(path, prompt)
    print(result)