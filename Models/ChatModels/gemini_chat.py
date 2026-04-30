from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import SecretStr
import os

load_dotenv()

raw_key = os.getenv('GOOGLE_API_KEY')
if not raw_key :
    print("❌ Error: GOOGLE_API_KEY not found! Check your .env file.")
else :
    api_key =  SecretStr(raw_key)  

    model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', 
                                api_key=api_key,
                                temperature=0.4, 
                                max_tokens=400)

    result = model.invoke("suggest me 5 good movies to watch No details just the names and capital of india ?")
    print(result.content)