from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-9b-it",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACE_API_KEY")
) 

model= ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(gt=18, description="The age of the person")
    email: str = Field(description="The email address of the person")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name, age and email of a fictional {place} person \n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.format(place="Indian") # you coulh have also used the template.invoke({"place": "Indian"}) to get the same result.
result = model.invoke(prompt)
final_result = parser.parse(result.content) # type: ignore
print("\nParsed Result:\n", final_result)
print(type(final_result))


# Using Chain to combine the steps and get the final result in one go along with the output parser to get the final result as a dictionary instead of a message object.
chain = template | model | parser
chain_result = chain.invoke({"place": "Indian"})
print("\nFinal Result:\n", chain_result)
print(type(chain_result))