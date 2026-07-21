import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "dummy"))
try:
    response = client.interactions.create(
        model="gemini-3.5-flash",
        prompt="Say hello world"
    )
    print("Success:", dir(response))
except Exception as e:
    print("Error:", e)
