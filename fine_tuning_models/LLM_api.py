# смотрите документацию https://api-docs.deepseek.com/

import os
from openai import OpenAI


api_key = os.environ.get('DEEPSEEK_API_KEY')
assert api_key, "DEEPSEEK_API_KEY не задан"
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {"role": "system", "content": """Роль: CV-разметчик. Задача: выдать YOLO-разметку для счетчика.
                                         Классы: 0=1, 1=2, 2=3, 3=4, 4=5, 5=6, 6=7, 7=8, 8=9, 9=0, 10=point, 11=T, 12=sigma/сумма.
                                         Формат: class_id x_center y_center width height (нормализованные 0-1).
                                         Правила: цифры — плотно по каждой; point — только разделитель; T — регистр; sigma — Σ.
                                         Вывод: только строки разметки, без пояснений."""},
        {"role": "user", "content": "Hello"},
    ],
    stream=False,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

print(response.choices[0].message.content)

key = "sk-751caa1d7f304060aef15693114a969b"