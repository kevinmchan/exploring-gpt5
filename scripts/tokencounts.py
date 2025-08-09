"""
A script to test basic features in the OpenAI Response API:
- Generating responses and getting the token usage
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI()


response = client.responses.create(
    model="gpt-5-nano", input=[{"role": "user", "content": "Hello, gpt-5!"}]
)
print(response.usage)
