from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "What is the capital of France?"

embedding = model.embed_query(text)

print(embedding)
print(len(embedding))