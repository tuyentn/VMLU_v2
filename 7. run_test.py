#%%
import os
import json
import tqdm
import pandas as pd
import time
import re
import requests

# #%%
# url = "http://localhost:1234/api/v0/chat/completions"

# headers = {
#     "Content-Type": "application/json"
# }

# data = {
#     "messages": [
#         {"role": "system", "content": "Always answer."},
#         {"role": "user", "content": "Hello, how are you?"}
#     ],
#     "temperature": 0.7,
#     "max_tokens": -1,
#     "stream": False
# }

# response = requests.post(url, headers=headers, data=json.dumps(data))

# # Extract just the content from the response
# try:
#     assistant_message = response.json()["choices"][0]["message"]["content"]
#     print(assistant_message)
# except KeyError:
#     print("Could not find content in the response. Full response:")
#     print(response.json())

#%%
from together import Together

client = Together(api_key="xxx")

def chat(message):
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        messages=[
            {
                    "role": "user",
                    "content": message
            }
        ],
        max_tokens=None,
        temperature=0.1,
        top_p=0.7,
        top_k=5,
        max_token=1,
        repetition_penalty=1,
        stop=["<｜end▁of▁sentence｜>"],
        _gl="1*1277gsu*_gcl_au*MTk0NjU4MjE3Mi4xNzQxOTQwNDg2*_ga*MjE0NDYzMDA4OC4xNzQxOTQwNDg1*_ga_BS43X21GZ2*MTc0MTk0MDQ4NS4xLjAuMTc0MTk0MDQ4NS4wLjAuMA..*_ga_BBHKJ5V8S0*MTc0MTk0MDQ4NS4xLjAuMTc0MTk0MDQ4NS4wLjAuMA..",
    )
    return response.choices[0].message.content

def deepseek(message):
    from openai import OpenAI

    client = OpenAI(
            base_url = "https://o70dyeq7m6ixlexu.us-east-1.aws.endpoints.huggingface.cloud/v1/",
            api_key = "xxx"
        )

    response = client.chat.completions.create(
        model="tgi",
        messages=[
        {
            "role": "user",
            "content": message
        }
    ],
        top_p=0.7,
        temperature=0.1,
        max_tokens=2000,
        stream=False,
        seed=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None
    )

    return response.choices[0].message.content

def find_last_occurrence(input_string):
    target_chars = {'A', 'B', 'C', 'D', 'E'}
    
    # Iterate through the string in reverse
    for i in range(len(input_string) - 1, -1, -1):
        if input_string[i] in target_chars:
            return input_string[i]
    
    # Return None if no character from the target set is found
    return None

#%%
def handle(start_idx):
    data = []
    with open('vlmu_mqa_v1.5/consolidated_data.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))

    # data = data[:1] # for debuging only
    all_res = []
    for idx, doc in enumerate(tqdm.tqdm(data[start_idx:start_idx+1000])):
        text_choice = '\n'.join(doc['choices'])
        prompt = "Kiến thức cơ bản: "+doc["example"]+"\n\nChỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau: \n" \
                + doc["question"] \
                + "\n\n" \
                + text_choice \
                + "\n" \
                + "Đáp án: " 
        
        response = deepseek(prompt)
        # print(response)
        # the response have many "Đáp án: " but get ans after the last "Đáp án: "
        answer = response[response.rfind('Đáp án: ')+8:response.rfind('Đáp án: ')+9]

        if answer not in ['A', 'B', 'C', 'D', 'E']:
            answer = find_last_occurrence(response)

        all_res.append({
            "id": doc['id'],
            "prompt": prompt,
            "question": doc["question"],
            "answer": answer
        })

    df = pd.DataFrame(all_res)
    df['answer'] = df.answer.map(lambda x: x[0].lower())
    df['answer'] = df['answer'].map(lambda x: re.sub(r'[^abcdef]', '', x))
    df[['id', 'answer']].to_csv('submission_'+str(start_idx)+'.csv', index=None)

#%%
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--Output", help = "Show Output")

    # Read arguments from command line
    args = parser.parse_args()

    handle(int(args.Output))

