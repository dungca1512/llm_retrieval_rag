from sentence_transformers import SentenceTransformer
import torch

# Download from the 🤗 Hub
model = SentenceTransformer("hiieu/halong_embedding")

# Define query and documents
query = "Bóng bàn có lợi ích gì cho sức khỏe?"
docs = [
    "Bóng đá giúp cải thiện sức khỏe tim mạch và tăng cường sức bền.",
    "Bóng đá là môn thể thao phổ biến nhất thế giới.",
    "Chơi bóng đá giúp giảm căng thẳng và cải thiện tâm lý.",
    "Bóng đá có thể giúp bạn kết nối với nhiều người hơn.",
    "Bóng đá không chỉ là môn thể thao mà còn là cách để giải trí.",
    "Ngoài bóng đá ra, còn nhiều môn thể thao khác cũng giúp cải thiện sức khỏe."
]

# Encode query and documents
query_embedding = model.encode([query])
doc_embeddings = model.encode(docs)
similarities = model.similarity(query_embedding, doc_embeddings).flatten()

# Sort documents by cosine similarity
sorted_indices = torch.argsort(similarities, descending=True)
sorted_docs = [docs[idx] for idx in sorted_indices]
sorted_scores = [similarities[idx].item() for idx in sorted_indices]

# Print sorted documents with their cosine scores
for doc, score in zip(sorted_docs, sorted_scores):
    print(f"Document: {doc} - Cosine Similarity: {score:.4f}")

# Document: Bóng đá giúp cải thiện sức khỏe tim mạch và tăng cường sức bền. - Cosine Similarity: 0.7318
# Document: Chơi bóng đá giúp giảm căng thẳng và cải thiện tâm lý. - Cosine Similarity: 0.6623
# Document: Bóng đá không chỉ là môn thể thao mà còn là cách để giải trí. - Cosine Similarity: 0.6102
# Document: Bóng đá có thể giúp bạn kết nối với nhiều người hơn. - Cosine Similarity: 0.4988
# Document: Bóng đá là môn thể thao phổ biến nhất thế giới. - Cosine Similarity: 0.4828
