import os
import PyPDF2
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import numpy as np


# Đọc file PDF
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


# Tạo embedding
def create_embedding(text, model):
    embedding = model.encode(text)
    return embedding


# Lưu vào MongoDB
def save_to_mongodb(file_path, text, embedding):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['pdf_database']
    collection = db['pdf_embeddings']

    document = {
        'file_path': file_path,
        'text': text,  # Lưu nội dung text
        'embedding': embedding.tolist()
    }

    result = collection.insert_one(document)
    print(f"Đã lưu embedding và text cho {file_path} với ID: {result.inserted_id}")

    client.close()


# Xử lý tất cả file PDF trong một thư mục
def process_directory(directory_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            try:
                pdf_text = read_pdf(file_path)
                pdf_embedding = create_embedding(pdf_text, model)
                save_to_mongodb(file_path, pdf_text, pdf_embedding)
                print(f"Đã xử lý thành công: {filename}")
            except Exception as e:
                print(f"Lỗi khi xử lý {filename}: {str(e)}")


# Sử dụng hàm
pdf_directory = '../data/papers'
process_directory(pdf_directory)
