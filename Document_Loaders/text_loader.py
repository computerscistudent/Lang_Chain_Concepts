from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = PromptTemplate(
    template = "Write a summary of the following poem:\n\n{poem}",
    input_variables = ["poem"],
)

parser = StrOutputParser()

loader = TextLoader(file_path="cricket.txt")

docs = loader.load()

# print(type(docs)) # list

# print(len(docs)) # 1

# print(docs[0]) # the actual document object

# print(type(docs[0])) # langchain_core.document_loaders.base.Document

# print(docs[0].page_content) # the actual text content of the document
# print(docs[0].metadata) # the metadata of the document
# print(docs[0].metadata["source"]) # the source of the document

chain = prompt | model | parser
rez  = chain.invoke({"poem": docs[0].page_content})
print(rez)

