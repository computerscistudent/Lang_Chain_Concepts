from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
import os

load_dotenv()

model1 = ChatOpenAI(model="gpt-4o-mini")
llm = HuggingFaceEndpoint(model="google/gemma-2-9b-it", task="text-generation", huggingfacehub_api_token= os.getenv("HUGGINGFACE_API_KEY"))

model2 = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template='Generate short and simple notes from the given text: {text}.',
    input_variables=['text']
)

template2 = PromptTemplate(
    template='Generate 5 short questions and answers from the following {text}.', 
    input_variables=['text']
)

template3 = PromptTemplate(
    template = 'Merge the following notes and quiz into a single text: {notes} {quiz}.',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes': template1 | model1 | parser,
        'quiz': template2 | model2 | parser
    }
)

merge_chain = template3 | model1 | parser

chain = parallel_chain | merge_chain

text = """"
    Technical Report: The Great AttractorSubject: Gravitational Anomalies & Large-Scale Galactic MotionTarget: Centaurus Supercluster / Laniakea Supercluster1. OverviewThe Great Attractor is a massive gravitational anomaly located in intergalactic space at the center of the Laniakea Supercluster. It serves as a central "gravity well" toward which our own Milky Way, along with millions of other galaxies, is being pulled.2. Physical CharacteristicsMass: Estimated to be roughly $10^{15}$ solar masses (equivalent to 100,000 Milky Ways).Distance: Approximately 150 to 250 million light-years away from Earth.Location: Situated in the direction of the constellations Centaurus and Norma.Velocity: The Milky Way and its neighbors are traveling toward this point at a staggering 600 kilometers per second.3. Key Scientific ChallengesThe "Zone of Avoidance"For decades, studying the Great Attractor was nearly impossible because it sits directly behind the galactic plane of the Milky Way. This region, known as the Zone of Avoidance, is filled with thick dust and gas that blocks visible light.Solution: Astronomers now use X-ray and Radio astronomy to "see through" the dust and map the structures hidden behind our own galaxy.Dark Matter AssociationThe observed visible matter (stars and galaxies) in the region does not have enough mass to explain the immense gravitational pull being exerted. This strongly suggests a high concentration of Dark Matter, which provides the extra "invisible" mass required to influence galactic motion on such a scale.4. Components of the AnomalyThe Great Attractor is not a single "object" like a black hole; rather, it is a massive concentration of several galaxy clusters:The Norma Cluster: The closest massive cluster near the heart of the anomaly.The Hydra and Centaurus Clusters: Major contributors to the overall gravitational field.The Shapley Supercluster: Research suggests that while we are moving toward the Great Attractor, both we and the Great Attractor are being pulled toward an even larger structure further away called the Shapley Supercluster.
"""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()

