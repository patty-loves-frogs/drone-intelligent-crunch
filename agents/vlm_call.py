import base64
from openai import OpenAI, OpenAIError

def VLM_Call(image_path, text_prompt):
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="sk-no-key-required")
        response = client.chat.completions.create(
            model="uwaaa",
            messages=[
                {
                    "role": "system",
                    "content" : "You are John Milton. You speak in Old English. Use emojis."
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
            max_tokens=512
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        status_code = getattr(e, 'status_code', 'N/A')
        return f"Error {status_code}: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

path = r"C:\Users\chouh\Pictures\Screenshots\Screenshot 2026-03-07 210231.png"
prompt = "What do you see in this image? Describe in 2 sentences max, with one line per sentence."

result = VLM_Call(path, prompt)
print(result)