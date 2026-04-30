from langchain_huggingface import HuggingFaceEmbeddings

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome."
]

embedding = model.embed_documents(text)

print(embedding)
print(len(embedding))
print("Length of each embedding vector:")
print(len(embedding[0]))