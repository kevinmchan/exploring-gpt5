"""
A script to test basic features in the OpenAI Response API:
- Generating responses as structured outputs using pydantic models
"""

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


load_dotenv()


client = OpenAI()


# I like OpenAI's example, so I'm going to copy it directly
class Step(BaseModel):
    explanation: str
    output: str


class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str


response = client.responses.parse(
    model="gpt-5-nano",
    input=[
        {
            "role": "system",
            "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
        },
        {"role": "user", "content": "how can I solve 8x + 7 = -23"},
    ],
    text_format=MathReasoning,
)

print(response.output_parsed)
