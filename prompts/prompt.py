import streamlit as st
from langchain.prompts import PromptTemplate, load_prompt
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.3)

st.header("wecome to my Research Tool!!")

selected_paper = st.selectbox("select the paper you want to summarize", ["Attention Is All You Need", 
                                                                      "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 
                                                                      "GPT-3: Language Models are Few-Shot Learners",
                                                                      "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
                                                                      "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
                                                                      "Other"])

if selected_paper == "Other":
    paper_input = st.text_input("Enter the full name of the research paper")
else :
    paper_input = selected_paper    

style_input = st.selectbox("select explanation style", ["Beginner-friendly", "Technical", "Code oriented", "Concise", "Mathematical"])

length_input = st.selectbox("select summary length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (Detailed explanation)"])


template = load_prompt("template.json")
prompt = template.invoke({
    'paper_input' : paper_input,
    'style_input' : style_input,
    'length_input' : length_input
})


if st.button("summarize") :
    st.write("summary -:")
    result = model.invoke(prompt)

    # Another way to do it is to create a chain with the template and the model, and then invoke the chain with the input variables. This can be useful if you want to reuse the same template with different models or if you want to add more steps to the chain in the future.

    # chain = template | model
    # result = chain.invoke({
    #     'paper_input' : paper_input,
    #     'style_input' : style_input,
    #     'length_input' : length_input
    # })
    
    st.write(result.content)