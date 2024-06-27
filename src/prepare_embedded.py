from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


# Load the BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class EmbeddedVector:

    def get_word_embedding(word):
        # Tokenize the input text
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        
        # Get embedding
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get embedding of first token (CLS token)
        word_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return word_embedding


    def cosine_similarity(v1, v2):
        return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0), dim=1).item()

word_1 = "pineapple"
word_2 = "banana"
embedding_1 = EmbeddedVector.get_word_embedding(word_1)
embedding_2 = EmbeddedVector.get_word_embedding(word_2)

# print(f"Embedding: {embedding}")
# print(f"Embedding shape: {embedding.shape}")

print(f"Cosine similarity between '{word_1}' and '{word_2}': {EmbeddedVector.cosine_similarity(embedding_1, embedding_2)}")
