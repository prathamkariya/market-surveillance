from google import genai
import inspect

client = genai.Client(api_key="dummy")
try:
    print(dir(client.interactions))
    print(inspect.signature(client.interactions.create))
except Exception as e:
    print("Error:", e)
