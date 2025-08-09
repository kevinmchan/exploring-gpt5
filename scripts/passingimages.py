import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.responses import (
    ResponseInputParam,
    ResponseInputItemParam,
    EasyInputMessageParam,
)

load_dotenv()


client = OpenAI()


image_path = Path(__file__).parent.parent / "assets/hello.jpeg"
with open(image_path, "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode("utf-8")


inputs: ResponseInputParam = [
    EasyInputMessageParam(
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What do you see?"},
                {
                    "detail": "auto",
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_data}",
                },
            ],
        }
    )
]

response = client.responses.create(
    model="gpt-5-nano",
    input=inputs,
)

print(response.output_text)
