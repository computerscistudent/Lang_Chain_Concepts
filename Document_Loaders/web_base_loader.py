from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template = " Answer the following {question} from the following text:\n\n{text}",
    input_variables = ["question", "text"],
)

parser = StrOutputParser()

url = "https://en.wikipedia.org/wiki/Dhurandhar:_The_Revenge"

loader = WebBaseLoader(url)

docs = loader.load()

# print(type(docs))
# print(len(docs))

# print(docs[0].page_content)

chain = prompt | model | parser
rez  = chain.invoke({"question": "What is the movie about and its lifetime collection?", "text": docs[0].page_content})
print(rez)