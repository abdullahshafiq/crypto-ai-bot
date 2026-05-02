import requests
r = requests.get("https://openrouter.ai/api/v1/models")
if r.status_code == 200:
    for m in r.json()["data"]:
        if "free" in m["id"]:
            print(m["id"])
