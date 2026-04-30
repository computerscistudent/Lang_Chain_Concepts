from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful  customer support agent."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}"),
])

chat_history = []
with open("prompts/chat_history.txt", "r") as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    "history": chat_history,
    "query": "I want to request a refund for my order #12345."
})

print(prompt)