import os
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Đường dẫn đến thư mục chứa các file markdown
markdown_folder = "./data/books"

# Hàm để đọc nội dung của các file markdown
def read_markdown_files(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                documents.append(file.read())
    return documents

# Đọc các file markdown
markdown_docs = read_markdown_files(markdown_folder)

# Chia nhỏ văn bản
text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.create_documents(markdown_docs)

# Khởi tạo model embedding
embeddings = HuggingFaceEmbeddings()

# Tạo và lưu trữ vector embeddings trong Chroma với tên collection cụ thể
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db", collection_name="markdown_collection")

# Lưu cơ sở dữ liệu
db.persist()

print("Embedding and saving done.")