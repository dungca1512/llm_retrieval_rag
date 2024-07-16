import requests
import json
import curl

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyAocGuPCObZsQGXA-ueI8Bv_CNl8gw5b7g"

payload = json.dumps({
  "contents": [
    {
      "parts": [
        {
          "text": "What is Retrieval-Augmented Generation?"
        }
      ]
    }
  ]
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(curl.parse(response))
