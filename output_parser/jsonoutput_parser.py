from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-9b-it",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACE_API_KEY")
) 

model= ChatHuggingFace(llm=llm)

json_parser = JsonOutputParser()
template = PromptTemplate(
    template="Give me the name,age and city of a fictional person \n {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": json_parser.get_format_instructions()}
)

prompt = template.format()

result = model.invoke(prompt)

# print("Prompt:\n", prompt)  
# print()
# print("Result:\n", result)
# # print(type(result))

print()
# the result is a message object, we need to get the content of the message and then parse it using the json parser to get the final result as a dictionary.
final_result = json_parser.parse(result.content) # type: ignore
print("\nParsed Result:\n", final_result)
print(type(final_result))
print("\nName:", final_result["name"])
print("Age:", final_result["age"])      
print("City:", final_result["city"])


# Using Chain to combine the steps and get the final result in one go along with the output parser to get the final result as a dictionary instead of a message object.

print("\nUsing Chain:\n")
chain = template | model | json_parser
chain_result = chain.invoke({})
print("\nFinal Result:\n", chain_result)
print(type(chain_result))
print("\nName:", chain_result["name"])
print("Age:", chain_result["age"])
print("City:", chain_result["city"])