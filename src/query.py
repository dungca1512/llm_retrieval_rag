from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import requests
import json
from dotenv import load_dotenv
import os
import numpy as np

# Load API key from file .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"

# Init SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # This model produces 384-dimensional embeddings

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['pdf_database']
collection = db['pdf_embeddings']


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_relevant_context(query, top_k=3, max_length=2000):
    # Create embedding for query
    query_embedding = model.encode(query)

    # Search for similar documents in MongoDB
    all_docs = list(collection.find())
    similarities = []
    for doc in all_docs:
        doc_embedding = np.array(doc['embedding'])
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append((similarity, doc))

    # Sort by similarity and get top_k results
    similarities.sort(key=lambda x: x[0], reverse=True)
    top_results = similarities[:top_k]

    # Combine similar paragraphs into one context
    context = "\n\n".join([doc['file_path'] for _, doc in top_results])
    metadatas = [{"file_path": doc['file_path']} for _, doc in top_results]

    # If the context is too long, cut it shorter
    if len(context) > max_length:
        context = context[:max_length] + "..."

    return context, metadatas


def answer_question(query):
    # Get context and metadata
    context, metadatas = get_relevant_context(query)

    # Generate prompt for Gemini
    prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, 
    say "I don't have enough information to answer that question."

    Context: {context}

    Question: {query}

    Answer: """

    # Call Google Gemini API
    payload = json.dumps({
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 1,
            "topP": 1,
            "maxOutputTokens": 2048,
        }
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(API_URL, headers=headers, data=payload)
    response_json = response.json()

    if 'candidates' in response_json and len(response_json['candidates']) > 0:
        return response_json['candidates'][0]['content']['parts'][0]['text'], context, metadatas
    else:
        error_message = response_json.get('error', {}).get('message', "Unknown error occurred")
        return f"Sorry, I couldn't generate a response. Error: {error_message}", context, metadatas


def main():
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        answer, context, metadatas = answer_question(query)
        print(f"Answer: {answer}\n")
        # Uncomment the following lines if you want to see context and metadata
        # print(f"Context used:\n{context}\n")
        # print("Metadata of documents containing the context:")
        # for metadata in metadatas:
        #     print(metadata)
        # print("\n")


if __name__ == "__main__":
    main()
