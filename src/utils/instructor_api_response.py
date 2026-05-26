from ai_sdk import generate_object, generate_text, openai
from dotenv import load_dotenv
from openai import OpenAI
from opencode_ai import Opencode
from pydantic import BaseModel

load_dotenv()
# client = OpenAI(base_url ='https://opencode.ai/zen/go/v1/chat/completions')
# # client= Opencode(base_url ='https://opencode.ai/zen/go/v1/chat/completions')
# # client.


# class CalendarEvent(BaseModel):
#     name: str
#     date: str
#     participants: list[str]
# response = client.responses.parse(
#     model="deepseek-v4-flash",
#     input=[
#         {"role": "system", "content": "Extract the event information."},
#         {
#             "role": "user",
#             "content": "Alice and Bob are going to a science fair on Friday.",
#         },
#     ],
#     text_format=CalendarEvent,
# )
# event = response.output_parsed
# print(event)

# model = openai(model = "opencode-go/deepseek-v4-flash")
class Person(BaseModel):
    name: str
    age: int



# # res = generate_object(
# #     model=model,
# #     schema=Person,
# #     prompt="Create a person named Alice, age 30"
# # )
# # print(res.object)  # Person(name='Alice', age=30)

# res = generate_text(model=model, prompt="Tell me a haiku about Python")
# print(res.text)
import instructor
from instructor.processing.multimodal import Image

client = instructor.from_provider("google/gemini-3-flash-preview",async_client=True)
# client = instructor.from_provider("openrouter/deepseek/deepseek-v4-flash",base_url="https://openrouter.ai/api/v1",async_client=True)


async def get_llm_response_from_instructor(user_input: str,
                                           response_format: BaseModel,
                                           image_url: str | None=None,
                                           system_prompt:str="You are a helpful assistant",
                                           max_tokens:int=1000):
    user_message= [user_input]
    if image_url is not None:
        user_message.append(Image.from_url(image_url))

    messages= [
        {   "role":"system",
            "content":system_prompt,
            "role": "user",
            "content": user_message
            # "role": "user",
            # "content": {"type": "image_url", "image_url": {"url": image_url}},
        }
    ]

    response = await client.create(
        messages=messages,
        response_model=response_format,
        generation_config={
            "temperature": 0,
            "max_tokens": max_tokens,
            "top_p": 1,
            "top_k": 32,
        },
    )
    return response
