from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from typing import List

load_dotenv()

model = ChatOpenAI(model='gpt-4')

# Option 1
chat : List[BaseMessage] = [SystemMessage(content="You are a helpful assistant.")] 
# option 2
# chat = []
# chat.append(SystemMessage(content="You are a helpful assistant."))

while True:
    user_input = input("You -: ")
    chat.append(HumanMessage(content=user_input))
    if user_input == "Leave".lower():
        break
    else:
        result = model.invoke(chat)
        chat.append(AIMessage(content=result.content))
        print(f"AI -: {result.content}")


print(chat)        