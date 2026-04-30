from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

model = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=16)

doc = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Italy is Rome."
]

embedding = model.embed_documents(doc)
print(embedding)
print(len(embedding))
print("Length of each embedding vector:")
print(len(embedding[0]))