import os
from dotenv import load_dotenv
import requests

load_dotenv()

print("Checking Gemini API...")
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
    payload = {
        "contents": [{"parts": [{"text": "Hello, answer in exactly one word: Success"}]}],
        "generationConfig": {"maxOutputTokens": 20, "temperature": 0.0}
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        print("  -> Raw status:", r.status_code)
        print("  -> Raw content:", r.text)
    except Exception as e:
        print(f"  -> Gemini API threw exception: {e}")
else:
    print("  -> GEMINI_API_KEY is not set.")

print("\nChecking OpenRouter API...")
or_key = os.getenv("OPENROUTER_API_KEY")
if or_key:
    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [{"role": "user", "content": "Hello, answer in exactly one word: Success"}],
        "max_tokens": 20,
        "temperature": 0.0
    }
    try:
        r = requests.post(url, json=payload, headers={
            "Authorization": f"Bearer {or_key}",
            "HTTP-Referer": "https://crypto-ai-bot.local",
            "Content-Type": "application/json"
        }, timeout=10)
        if r.status_code == 200:
            print("  -> OpenRouter API is working! Response:", r.json()["choices"][0]["message"]["content"].strip())
        else:
            print(f"  -> OpenRouter API failed with status {r.status_code}: {r.text}")
    except Exception as e:
        print(f"  -> OpenRouter API threw exception: {e}")
else:
    print("  -> OPENROUTER_API_KEY is not set.")

print("\nChecking OpenAI API...")
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    from openai import OpenAI
    try:
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, answer in exactly one word: Success"}],
            max_tokens=20,
            temperature=0.0
        )
        print("  -> OpenAI API is working! Response:", response.choices[0].message.content.strip())
    except Exception as e:
        print(f"  -> OpenAI API threw exception: {e}")
else:
    print("  -> OPENAI_API_KEY is not set.")
