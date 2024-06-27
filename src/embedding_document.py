import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import os


# Bước 1 & 2: Đọc file PDF và trích xuất văn bản
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


# Bước 3: Tạo embedding cho văn bản
def create_embedding(text, model):
    # Chia văn bản thành các đoạn
    paragraphs = text.split('\n\n')
    embeddings = model.encode(paragraphs)
    return embeddings, paragraphs


# Bước 4: Lưu vào Chroma database
def save_to_chroma(pdf_name, embeddings, paragraphs):
    client = chromadb.PersistentClient(path="./chroma_db")

    # Tạo hoặc lấy collection
    collection = client.get_or_create_collection(name="paper_embeddings")

    # Tạo danh sách các ID duy nhất
    ids = [str(uuid.uuid4()) for _ in paragraphs]

    # Thêm dữ liệu vào collection
    collection.add(
        embeddings=embeddings.tolist(),
        documents=paragraphs,
        metadatas=[{"pdf_name": pdf_name} for _ in paragraphs],
        ids=ids
    )


# Hàm để xử lý tất cả các file PDF trong một thư mục
def process_pdf_folder(folder_path, model):
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")
            
            text = extract_text_from_pdf(pdf_path)
            embeddings, paragraphs = create_embedding(text, model)
            save_to_chroma(filename, embeddings, paragraphs)
            
            print(f"Completed processing {filename}")


# Hàm để tìm kiếm đoạn văn tương tự
def search_similar_paragraphs(query, top_k=5):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("paper_embeddings")

    # Tạo embedding cho câu query
    query_embedding = model.encode([query])[0]

    # Tìm kiếm các đoạn văn tương tự
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    return results

# Sử dụng các hàm
pdf_folder = './data'  # Đường dẫn đến thư mục chứa các file PDF

# Tải mô hình Sentence Transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Xử lý tất cả các file PDF trong thư mục
process_pdf_folder(pdf_folder, model)

print("Embedding completed and saved to Chroma database for all PDF files.")


# # Ví dụ sử dụng hàm tìm kiếm
# query = "Retrieval-Augmented Generation"
# similar_paragraphs = search_similar_paragraphs(query)

# for i, (document, metadata, distance) in enumerate(zip(similar_paragraphs['documents'][0],
#                                                        similar_paragraphs['metadatas'][0],
#                                                        similar_paragraphs['distances'][0])):
#     print(f"Result {i + 1}:")
#     print(f"Similarity: {1 - distance:.4f}")
#     print(f"PDF: {metadata['pdf_name']}")
#     print(f"Paragraph: {document[:100]}...")
#     print()
