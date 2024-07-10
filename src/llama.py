# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

print(pipe("Write me a poem about Machine Learning.")['generated_text'])
