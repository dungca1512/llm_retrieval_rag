from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from config import Config
from rag import RAG

load_dotenv()

config = Config()

url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.get_llm_model()}:generateContent?key={config.get_gemini_key()}"

model = SentenceTransformer(config.get_embedding_model())  # This model produces 384-dimensional embeddings

client = MongoClient(config.get_db_uri())
db = client[config.get_db_name()]
collection = db[config.get_db_collection()]


def main():
    rag = RAG(url=url, client=client, model=model, db=db, collection=collection)
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        answer, context, metadatas = rag.answer_question(query)
        print(f"Answer: {answer}\n")
        print(f"Context: {context}\n")
        print(f"Metadata: {metadatas}\n")


if __name__ == "__main__":
    main()
