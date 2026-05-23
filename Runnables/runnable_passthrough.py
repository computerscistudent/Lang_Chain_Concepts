from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough , RunnableSequence
import os

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="explain the following joke {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain1 = RunnableSequence(prompt1 , model , parser)

chain2 = RunnableParallel({
        'Joke' : RunnablePassthrough(),
        'explaination' : RunnableSequence(prompt2,model,parser)
    }
)

final_chain = RunnableSequence(chain1,chain2)

result = final_chain.invoke({'topic':'Cricket'})

# print(f"Joke -: {result['Joke']}")
# print()
# print(f"Meaning -: {result['explaination']}")

rez = """ Joke -: {}\n Meaning -: {} """.format(result['Joke'],result['explaination'])

print(rez)