import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
from dotenv import load_dotenv
import os

# Load API key from file .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

API_URL = f"https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key={API_KEY}"

# Init Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to Chroma database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("paper_embeddings")


def get_relevant_context(query, top_k=3, max_length=500):
    # Create embedding for query
    query_embedding = model.encode(query).tolist()

    # Search for similar passages
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Combine similar paragraphs into one context
    context = "\n\n".join(results['documents'][0])

    # If the context is too long, cut it shorter
    if len(context) > max_length:
        context = context[:max_length] + "..."

    return context


def answer_question(query):
    # Get context
    context = get_relevant_context(query)

    # Generate prompt for PaLM
    prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, 
    say "I don't have enough information to answer that question."
    
    Context: {context}

    Question: {query}

    Answer:"""

    # Call Google PaLM API
    payload = json.dumps({
        "prompt": {
            "text": prompt
        },
        "temperature": 0.7,
        "candidate_count": 1
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(API_URL, headers=headers, data=payload)
    response_json = response.json()

    if 'candidates' in response_json and len(response_json['candidates']) > 0:
        return response_json['candidates'][0]['output']
    else:
        return "Sorry, I couldn't generate a response."

def main():
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        answer = answer_question(query)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
