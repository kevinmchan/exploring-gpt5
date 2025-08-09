"""
A script to test basic features in the OpenAI Response API:
- How to manage the model inputs when using tools
"""

import json
import pprint

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseInputParam,
    ToolParam,
)
from openai.types.responses.response_input_param import FunctionCallOutput

load_dotenv()


client = OpenAI()


# define tools
orders_data = {
    "abc": "c123",
    "def": "c456",
}


def get_orders():
    return str(list(orders_data.keys()))


def get_customer_by_order_id(order_id: str):
    return orders_data.get(order_id)


def call_tool(response: ResponseFunctionToolCall):
    tool = tool_mapping.get(response.name)
    if tool:
        try:
            args = json.loads(response.arguments)
            result = tool(**args)
        except Exception as e:
            result = f"Tool call failed with error:\n{e}"
    else:
        result = f"No tool with name `{response.name}` available."

    tool_output = FunctionCallOutput(
        {
            "type": "function_call_output",
            "call_id": response.call_id,
            "output": result,
        }
    )
    return tool_output


tool_mapping = {
    "get_orders": get_orders,
    "get_customer_by_order_id": get_customer_by_order_id,
}

tools: list[ToolParam] = [
    {
        "type": "function",
        "name": "get_orders",
        "description": "Returns a list ids for current open orders.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_customer_by_order_id",
        "description": "Returns a customer id for a given order id.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "Identifier for order to lookup",
                },
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# We'll provide an initial prompt and get response
# We'll look for tool invokations, attempt to run tool and send back tool output
# We'll repeat this for as long as we get a tool call invokation
context: ResponseInputParam = [
    {
        "role": "user",
        "content": "I'd like to know the customer ids for all customers with an open order.",
    },
]
while True:
    response = client.responses.create(
        model="gpt-5-nano",
        input=context,
        tools=tools,
        parallel_tool_calls=True,
        tool_choice="auto",
    )
    context += response.output
    tool_outputs = [
        call_tool(el) for el in response.output if el.type == "function_call"
    ]
    if tool_outputs:
        context += tool_outputs
    else:
        break

pprint.pp(context)
