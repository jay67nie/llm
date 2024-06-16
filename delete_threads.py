import time

import requests as req
from alive_progress import alive_it
from openai import OpenAI

token = "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"
org = "Personal"
url = "https://api.openai.com/v1/threads"

headers = {
    "Authorization": f"Bearer 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",
    "Openai-Organization": f"{org}",
    "OpenAI-Beta": "assistants=v1"
}
params = {"limit": 100}
resp = req.get(url, headers=headers, params=params)
print("id: ", resp)
ids = [t['id'] for t in resp.json()['data']]

client = OpenAI()

while len(ids) > 0:
    for tid in alive_it(ids, force_tty=True):
        client.beta.threads.delete(tid)
        time.sleep(1)
    resp = req.get(url, headers=headers, params=params)
    ids = [t['id'] for t in resp.json()['data']]
