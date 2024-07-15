from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client.llm_retrieval_rag
collection = db.data

# Load mô hình SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Hàm tạo embedding cho văn bản
def embed_text(text):
    return model.encode(text)

# Tạo embedding cho các document và lưu vào MongoDB
for doc in collection.find():
    content = doc.get('content', '') 
    if content:
        embedding = embed_text(content)
        collection.update_one({'_id': doc['_id']}, {'$set': {'embedding': embedding.tolist()}})

# Tạo danh sách embedding và doc_ids
embeddings = []
doc_ids = []

for doc in collection.find({'embedding': {'$exists': True}}):
    embeddings.append(doc['embedding'])
    doc_ids.append(doc['_id'])

embeddings = np.array(embeddings).astype('float32')

# Tạo FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Lưu FAISS index vào file (tùy chọn)
faiss.write_index(index, 'faiss_index.index')

print("Indexing completed!")

