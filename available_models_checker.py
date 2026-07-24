from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Get key from .env file
key = os.getenv("GROQ_API_KEY", "").strip('"\' ')

client = Groq(api_key=key)

print("Fetching active models from Groq API...\n")
response = client.models.list()

for model in response.data:
    if getattr(model, "active", True):
        print(f"✅ Model ID: {model.id:<30} | Owned By: {model.owned_by}")
