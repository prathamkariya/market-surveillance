import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "dummy"))
try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="Say hello world in python"
    )
    print("Success:", response.text)
except Exception as e:
    print("Error:", e)
