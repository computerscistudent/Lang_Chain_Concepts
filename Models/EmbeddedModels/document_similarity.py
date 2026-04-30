from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

model = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=16)

doc = [
    "Virat Kohli is a great batsman.",
    "Sachin Tendulkar is a legendary cricketer.",
    "Jay Shah is the president of BCCI.",
    "Dhoni is a former Indian cricket team captain."
]

query = "Who is the president of BCCI?"

embedding_doc = model.embed_documents(doc)
embedding_query = model.embed_query(query)

result = cosine_similarity([embedding_query], embedding_doc) #type: ignore

print("Cosine Similarity Scores:")
print(result)

for i in range(len(result[0])):
    print(f"Document: {doc[i]} - Similarity Score: {result[0][i]}")


print("\nMost Similar Document:")
most_similar_index = result[0].argmax()
print(f"{doc[most_similar_index]} - Similarity Score: {result[0][most_similar_index]}")


print("Highest Similarity Score and its Index:")
print(sorted(list(enumerate(result[0])), key=lambda x: x[1], reverse=True)[0])
print("Lowest Similarity Score and its Index:")
print(sorted(list(enumerate(result[0])), key=lambda x: x[1], reverse=True)[-1]) # you can also do print(sorted(list(enumerate(result[0])), key=lambda x: x[1], reverse=False)[0])