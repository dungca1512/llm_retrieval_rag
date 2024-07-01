import requests
import json
from os import getenv
from dotenv import load_dotenv

load_dotenv()

API_KEY = getenv("API_KEY")

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"

text = input("Enter your question: ")

payload = json.dumps({
  "contents": [
    {
      "parts": [
        {
          "text": text
        }
      ]
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request(method="POST", url=url, headers=headers, data=payload)
result = response.json()

print(result['candidates'][0]['content']['parts'][0]['text'])
