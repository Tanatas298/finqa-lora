from openai import OpenAI

from finqa_ft.data import load_split
from finqa_ft.program import execute
from finqa_ft.prompts import extract_program, messages

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")
ex = load_split("test")[0]
reply = client.chat.completions.create(
    model="finqa-4b",
    messages=messages(ex),
    temperature=0,
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
program = extract_program(reply.choices[0].message.content)
print(ex.question, "->", program, "=", execute(program, ex.table), "| gold:", ex.answer)
