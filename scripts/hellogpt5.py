"""
A script to test basic features in the OpenAI Response API:
- Generating responses (hello gpt5)
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI()

# Get a simple hello gpt-5 response
response = client.responses.create(
    model="gpt-5-nano", input=[{"role": "user", "content": "Hello, gpt-5!"}]
)
print(response.output_text)
