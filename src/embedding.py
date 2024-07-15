from sentence_transformers import SentenceTransformer
from chunking import process_pdfs_in_directory


def generate_embeddings(sections_list, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings_with_content = []

    for sections in sections_list:
        for section, content in sections.items():
            embedding = model.encode(content).tolist()  # Chuyển đổi embedding thành danh sách để lưu vào MongoDB
            embeddings_with_content.append({
                'section': section,
                'content': content,
                'embedding': embedding
            })
    return embeddings_with_content


if __name__ == "__main__":
    # Thay đổi đường dẫn tới thư mục chứa các file PDF của bạn
    directory_path = '../data/papers'
    all_sections = process_pdfs_in_directory(directory_path)

    # Tạo embedding cho từng phần tử trong danh sách
    embeddings = generate_embeddings(all_sections)

    # In ra embedding của từng phần tử
    print(embeddings)
