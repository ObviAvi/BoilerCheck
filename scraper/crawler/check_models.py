from google import genai
import os
from pathlib import Path
from dotenv import load_dotenv

# .env lives in the BoilerCheck repo root (parents[2]).
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

for m in client.models.list():
    print(m.name)