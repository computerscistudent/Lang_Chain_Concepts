from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.4, max_completion_tokens=40)

result = model.invoke("suggest me 5 good movies to watch No details just the names")
print(result.content)