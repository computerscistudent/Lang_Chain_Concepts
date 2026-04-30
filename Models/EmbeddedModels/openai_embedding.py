from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

model = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=32)

query = "What is the capital of France?"

embedding = model.embed_query(query)
print(embedding)
print(len(embedding))