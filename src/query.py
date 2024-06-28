import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import json
from dotenv import load_dotenv
import os

# Load API key from file .env
load_dotenv()
API_KEY = os.getenv("API_KEY")

API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}"

# Init HuggingFaceEmbeddings model
embeddings = HuggingFaceEmbeddings()

# Connect to Chroma database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("markdown_collection")  # Đảm bảo tên này khớp với tên collection khi bạn tạo embedding

def get_relevant_context(query, top_k=1, max_length=1000):
    # Create embedding for query
    query_embedding = embeddings.embed_query(query)

    # Search for similar query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    # Combine similar paragraphs into one context
    context = "\n\n".join(results['documents'][0])

    # If the context is too long, cut it shorter
    if len(context) > max_length:
        context = context[:max_length] + "..."

    return context, results['metadatas'][0]


def answer_question(query):
    # Get context and metadata
    context, metadatas = get_relevant_context(query)

    # Generate prompt for Gemini
    prompt = f"""Based on the following context from markdown documents, please answer the question. If the answer is not in the context, 
    say "I don't have enough information to answer that question."
    
    Context: {context}

    Question: {query}

    Answer:"""

    # Call Google Gemini API
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
        return response_json['candidates'][0]['content']['parts'][0]['text'], context, metadatas
    else:
        return "Sorry, I couldn't generate a response.", context, metadatas


def main():
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        answer, context, metadatas = answer_question(query)
        print(f"Answer: {answer}\n")
        print(f"Context used:\n{context}\n")
        print("Metadata of documents containing the context:")
        for metadata in metadatas:
            print(metadata)
        print("\n")


if __name__ == "__main__":
    main()
    