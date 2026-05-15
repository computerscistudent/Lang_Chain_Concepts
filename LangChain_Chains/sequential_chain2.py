from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

template1 = PromptTemplate(
    template='Generate a detailed report about {topic}.',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following {text}.',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'The Great Attractor'})

print(result)

chain.get_graph().print_ascii()