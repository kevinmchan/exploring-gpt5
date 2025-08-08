"""
A script to test basic features in the OpenAI Response API:
- Setting the reasoning effort
- Getting back the reasoning summary
"""

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ResponseReasoningItem

load_dotenv()


client = OpenAI()

# Set the reasoning effort and retrieve the reasoning summary
print("High reasoning effort:")
response = client.responses.create(
    model="gpt-5-nano",
    reasoning={"effort": "high", "summary": "detailed"},
    input=[{"role": "user", "content": "Hello, gpt-5!"}],
)
for el in response.output:
    if isinstance(el, ResponseReasoningItem):
        print("Summary:")
        print(el)
        break
    raise Exception("Did not find a reasoning item")
print(f"Response: {response.output_text}")


print("Low reasoning effort:")
response = client.responses.create(
    model="gpt-5-nano",
    reasoning={"effort": "low", "summary": "detailed"},
    input=[{"role": "user", "content": "Hello, gpt-5!"}],
)
for el in response.output:
    if isinstance(el, ResponseReasoningItem):
        print("Summary:")
        print(el)
        break
    raise Exception("Did not find a reasoning item")
print(f"Response: {response.output_text}")
