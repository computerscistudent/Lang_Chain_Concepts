from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough , RunnableSequence
import os

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="generate a linkdin post about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1,model,parser),
    'linkdin' : RunnableSequence(prompt2,model,parser) 
})

rez = parallel_chain.invoke({'topic':'AI'})

final_rez = """ Tweet -: {} \n LinkdinPost -: {} """.format(rez['tweet'], rez['linkdin'])
print(final_rez)
parallel_chain.get_graph().print_ascii()