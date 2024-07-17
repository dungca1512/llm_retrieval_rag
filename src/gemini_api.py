import requests
import json
from config import Config

config = Config()

url = f"https://generativelanguage.googleapis.com:443/v1beta/models/{config.get_llm_model()}:generateContent?key={config.get_gemini_key()}"

query = input("Enter your question: ")

payload = json.dumps({
  "contents": [
    {
      "parts": [
        {
          "text": query
        }
      ]
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.json()['candidates'][0]['content']['parts'][0]['text'])
