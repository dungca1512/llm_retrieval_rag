import requests
import json
from dotenv import load_dotenv
import os

# Load API key from file .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

API_URL = f"https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key={API_KEY}"

headers = {
    'Content-Type': 'application/json'
}


def query(prompt):
    payload = json.dumps({
        "prompt": {
            "text": prompt
        }
    })
    response = requests.post(API_URL, headers=headers, data=payload)
    return response.json()


def chatbot():
    print("Chatbot: Hello! I'm a simple chatbot. You can start chatting. Type 'quit' to exit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye! See you next time.")
            break

        response = query(user_input)

        if 'candidates' in response and len(response['candidates']) > 0:
            bot_response = response['candidates'][0]['output']
            print(f"Chatbot: {bot_response}")
        else:
            print("Chatbot: Sorry, I'm having trouble processing the response. Please try again.")
            print("Debug - API response:", response)


if __name__ == "__main__":
    chatbot()
