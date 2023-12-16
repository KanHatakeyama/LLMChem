import openai
import json

#ask gpt
def json_generate(prompt,model="gpt-3.5-turbo-1106"):
    response = openai.ChatCompletion.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "Return JSON"
        },
        {
            "role": "user",
            "content": f"""{prompt}"""
        }  
    ],
    response_format={ "type": "json_object" }
    )

    return (json.loads(response.choices[0]["message"]["content"]))


