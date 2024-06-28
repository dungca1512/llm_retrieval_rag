import requests

API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "What is Retrieval-Augmented Generation?",
})

print("Text: ", output[0]['generated_text'])