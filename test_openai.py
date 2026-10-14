import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

response = client.responses.create(
    model=model,
    input="Reply with exactly: OpenAI connection successful",
)

print(response.output_text)
