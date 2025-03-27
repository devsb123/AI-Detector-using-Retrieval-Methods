#import os
from openai import OpenAI
client = OpenAI()
    #api_key = os.environ.get("OPENAI_API_KEY"),
#)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn."
        }
    ]
)

print(completion.choices[0].message.content)