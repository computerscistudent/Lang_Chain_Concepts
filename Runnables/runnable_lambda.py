from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough , RunnableSequence, RunnableLambda
import os

load_dotenv()

def word_count(text):
    return len(text.split())

# runnable_word_counter = RunnableLambda(word_count)

# print(runnable_word_counter.invoke('Hi how are u'))

model = ChatOpenAI(model="gpt-4o-mini")

prompt1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_count) # you could also do RunnableLambda(lambda x : len(x.split()))
})

chain = joke_chain | parallel_chain

rez = chain.invoke({'topic':'AI'})

final_rez = """ Joke -: {} \n\n\n WordCount -: {} """.format(rez['joke'], rez['word_count'])

print(final_rez)