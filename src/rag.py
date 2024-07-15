import requests
import json
import numpy as np


class RAG:

    def __init__(self, url, model, client, db, collection) -> None:
        self.url = url
        self.model = model
        self.client = client
        self.db = db
        self.collection = collection

    @staticmethod
    def cosine_similarity(a, b):
        a = a.flatten()
        b = b.flatten()
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_relevant_context(self, query, top_k=5, max_length=2000):
        query_embedding = self.model.encode(query)

        all_docs = list(self.collection.find())
        similarities = []
        for doc in all_docs:
            doc_embedding = np.array(doc['embedding'])
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]

        context = "\n\n".join([doc['content'] for _, doc in top_results])
        metadatas = [
            {
                "section": doc['section'],
                "similarity": float(similarity)  # Convert to float for JSON serialization
            }
            for similarity, doc in top_results
        ]

        if len(context) > max_length:
            context = context[:max_length] + "..."

        return context, metadatas

    def answer_question(self, query):
        context, metadatas = self.get_relevant_context(query)

        prompt = f"""Based on the following context, please answer the question. If the answer is not in the context, 
        say "I don't have enough information to answer that question."

        Context: {context}

        Question: {query}

        Answer: """

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
        response = requests.post(self.url, headers=headers, data=payload)
        response_json = response.json()

        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            return response_json['candidates'][0]['content']['parts'][0]['text'], context, metadatas
        else:
            error_message = response_json.get('error', {}).get('message', "Unknown error occurred")
            return f"Sorry, I couldn't generate a response. Error: {error_message}"
