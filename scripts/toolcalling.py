"""
A script to test basic features in the OpenAI Response API:
- Tool calling in various scenarios:
    - calling exactly one tool
    - calling multiple tools in parallel
    - calling any number of tools (zero, one, many)
"""

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import ToolParam

load_dotenv()


client = OpenAI()

# define schemas for tools
tools: list[ToolParam] = [
    {
        "type": "function",
        "name": "first_tool",
        "description": "This is a dummy function call that repeats what you input.",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "text to repeat",
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "second_tool",
        "description": "This is a dummy function call that repeats what you input.",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "text to repeat",
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# Calling exactly one tool
# To demonstrate that the model is being forced into making a tool call,
# we won't actually prompt it to make a tool call
response = client.responses.create(
    model="gpt-5-nano",
    input=[{"role": "user", "content": "Hello."}],
    tools=tools,
    parallel_tool_calls=False,
    tool_choice="required",
)
print(response)
print(
    f"Received {len(response.output)} elements of classes {[el.__class__.__name__ for el in response.output]}"
)


# Calling multiple tools
response = client.responses.create(
    model="gpt-5-nano",
    input=[{"role": "user", "content": "Call `first_tool` and `second_tool`"}],
    tools=tools,
    parallel_tool_calls=True,
    tool_choice="required",
)
print(response)
print(
    f"Received {len(response.output)} elements of classes {[el.__class__.__name__ for el in response.output]}"
)


# Calling any number of tools
# Demonstrate that the model may choose not to call a tool
response = client.responses.create(
    model="gpt-5-nano",
    input=[{"role": "user", "content": "Hello. Don't call any tools."}],
    tools=tools,
    parallel_tool_calls=True,
    tool_choice="auto",
)
print(response)
print(
    f"Received {len(response.output)} elements of classes {[el.__class__.__name__ for el in response.output]}"
)
