from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-9b-it",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACE_API_KEY")
)

model= ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template = "write a detailed report on {topic}",
    input_variables = ["topic"]
)

template2 = PromptTemplate(
    template = "write a 5 line summary  on given text \n {text}",
    input_variables = ["text"]
)

# generic way to get the result of each step and print it out.
prompt1 = template1.invoke({"topic": "Black holes"})

result1 = model.invoke(prompt1)  

prompt2 = template2.invoke({"text": result1})

result2 = model.invoke(prompt2)

print("Detailed Report:\n", result1.content)
print("\nSummary:\n", result2.content)


print()

# Using Chain to combine the steps and get the final result in one go along with the output parser to get the final result as a string instead of a message object.
print("\nUsing Chain:\n")
parser = StrOutputParser()

chain  = template1 | model | parser | template2  | model | parser

result = chain.invoke({"topic": "Black holes"})

print("\nFinal Result:\n", result)