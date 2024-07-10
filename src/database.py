import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Đường dẫn đến thư mục chứa dữ liệu
data_folder = "./data"


def read_files(folder_path):
    documents = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif file.endswith(".md"):
                    loader = UnstructuredMarkdownLoader(file_path)
                    documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return documents


# Đọc các file
docs = read_files(data_folder)

# Chia nhỏ văn bản
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# Khởi tạo model embedding
embeddings = HuggingFaceEmbeddings()

# Tạo và lưu trữ vector embeddings trong Chroma với tên collection cụ thể
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db",
                           collection_name="all_documents_collection")

# Lưu cơ sở dữ liệu
db.persist()

print("Embedding and saving done.")
print(f"Total documents processed: {len(texts)}")
