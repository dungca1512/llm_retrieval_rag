from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_text(text):
  """Mã hóa văn bản thành embedding BERT."""
  encoded_input = tokenizer(text, return_tensors='pt')
  output = model(**encoded_input)
  return output.last_hidden_state

text = """
Samsung A12 là một chiếc smartphone giá rẻ 
nhưng sở hữu cấu hình ổn định cùng với viên pin 5000mAh
cho thời lượng suốt ngày dài. Bên cạnh đó điện thoại 
cũng sở hữu thiết kế thời trang và phù hợp với xu hướng.
"""

# Mã hóa văn bản
encoded_text = encode_text(text)

# In embedding
print(encoded_text)
print(encoded_text.shape)
