from pymongo import MongoClient
from embedding import generate_embeddings
from chunking import process_pdfs_in_directory


def save_to_mongodb(directory_path):

    client = MongoClient('mongodb://localhost:27017/')
    db = client['pdf_database']
    collection = db['pdf_embeddings']

    all_sections = process_pdfs_in_directory(directory_path)

    # Tạo embedding cho từng phần tử trong danh sách
    embeddings = generate_embeddings(all_sections)

    for emb in embeddings:
        collection.insert_one(emb)

    client.close()


if __name__ == "__main__":
    # Thay đổi đường dẫn tới thư mục chứa các file PDF của bạn
    directory_path = '../data/papers'
    save_to_mongodb(directory_path)
    print("Done!")
