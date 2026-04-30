from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    model="google/gemma-2-9b-it",
    task="text-generation",
    huggingfacehub_api_token= os.getenv("HUGGINGFACE_API_KEY")
) 

model= ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact1", description= "fact 1 about the topic"),
    ResponseSchema(name="fact2", description= "fact 2 about the topic"),
    ResponseSchema(name="fact3", description= "fact 3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about the {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.format(topic="moon") # you coulh have also used the template.invoke({"topic": "moon"}) to get the same result.

result = model.invoke(prompt)

print(result.content) # the result is a message object, we need to get the content of the message and then parse it using the structured output parser to get the final result as a dictionary.
print(type(result))

# Another way to get the result is parsing the result directly using the parser without getting the content of the message object, the parser will handle it internally and give us the final result as a dictionary.
final_result = parser.parse(result.content) # type: ignore
print("\nParsed Result:\n", final_result)
print(type(final_result))

# Doing the same thing using Chain to combine the steps and get the final result in one go along with the output parser to get the final result as a dictionary instead of a message object.

chain = template | model | parser

chain_result = chain.invoke({"topic": "moon"})
print("\nFinal Result:\n", chain_result)
print(type(chain_result))