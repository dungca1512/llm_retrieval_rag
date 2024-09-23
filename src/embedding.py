from sentence_transformers import SentenceTransformer
from chunking import process_pdfs_in_directory
from config import Config

config = Config()


def generate_embeddings(sections_list, model_name=config.get_embedding_model()):
    model = SentenceTransformer(model_name)
    embeddings_with_content = []

    for sections in sections_list:
        for section, content in sections.items():
            embedding = model.encode(content).tolist()  # Convert embedding to list and save to MongoDB
            embeddings_with_content.append({
                'section': section,
                'content': content,
                'embedding': embedding
            })
    return embeddings_with_content


if __name__ == "__main__":
    # Change path to data folder 
    directory_path = '../data/papers'
    all_sections = process_pdfs_in_directory(directory_path)

    # Embedding for each element in the list
    embeddings = generate_embeddings(all_sections)
    print(embeddings)
