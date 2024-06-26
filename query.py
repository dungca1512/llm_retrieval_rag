import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

# Thiết lập Google PaLM API key
API_KEY = "AIzaSyCRstOEDvR7AwoVfHeAzrJzGgNOw7HuXrk"  # Thay thế bằng API key của bạn nếu khác
API_URL = f"https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key={API_KEY}"

# Khởi tạo Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Kết nối tới Chroma database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("paper_embeddings")


def get_relevant_context(query, top_k=3):
    # Tạo embedding cho câu query
    query_embedding = model.encode(query).tolist()

    # Tìm kiếm các đoạn văn tương tự
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # Kết hợp các đoạn văn tương tự thành một context
    context = "\n\n".join(results['documents'][0])
    return context


def answer_question(query):
    # Lấy context liên quan
    context = get_relevant_context(query)

    # Tạo prompt cho PaLM
    prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, 
    say "I don't have enough information to answer that question."

Context: {context}

Question: {query}

Answer:"""

    # Gọi Google PaLM API
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


# Sử dụng hệ thống
while True:
    query = input("Enter your question (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    answer = answer_question(query)
    print(f"Answer: {answer}\n")
