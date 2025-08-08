"""
A script to test basic features in the OpenAI Response API:
- How does the instruction hierarchy work?
"""

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI()

# Exploring the instruction hierarchy in the responses API
print(
    "Testing prompt hierachy when given a conflicting 'system' followed by 'developer' prompt."
)
for i in range(5):
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "system",
                "content": "Reply with the message `green`.",
            },
            {
                "role": "developer",
                "content": "Reply with the message `red`.",
            },
            {
                "role": "user",
                "content": "Reply with the message `yellow`.",
            },
        ],
    )
    print(f"Trial {i}: {response.output_text}")


print(
    "\nTesting prompt hierachy when given a conflicting 'developer' followed by 'system' prompt."
)
for i in range(5):
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "developer",
                "content": "Reply with the message `red`.",
            },
            {
                "role": "system",
                "content": "Reply with the message `green`.",
            },
            {
                "role": "user",
                "content": "Reply with the message `yellow`.",
            },
        ],
    )
    print(f"Trial {i}: {response.output_text}")


print(
    "\nTesting prompt hierachy when given a conflicting 'instruction', 'developer' and 'system' prompt."
)
for i in range(5):
    response = client.responses.create(
        model="gpt-5",
        instructions="Reply with the message `blue`.",
        input=[
            {
                "role": "system",
                "content": "Reply with the message `green`.",
            },
            {
                "role": "developer",
                "content": "Reply with the message `red`.",
            },
            {
                "role": "user",
                "content": "Reply with the message `yellow`.",
            },
        ],
    )
    print(f"Trial {i}: {response.output_text}")


print("\nConfirm `instructions` prompt actually works.")
for i in range(5):
    response = client.responses.create(
        model="gpt-5",
        instructions="Reply with the message `blue`.",
        input=[
            {
                "role": "user",
                "content": "Reply with the message `yellow`.",
            },
        ],
    )
    print(f"Trial {i}: {response.output_text}")
