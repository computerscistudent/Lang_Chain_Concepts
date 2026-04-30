from langchain_openai import ChatOpenAI
from dotenv import load_dotenv 
from typing import TypedDict , Annotated, Literal

load_dotenv()  # Load environment variables from .env file

model = ChatOpenAI(model="gpt-4o-mini")

class Review(TypedDict):
    summary : str
    sentiment : str

class Review2(TypedDict):
    summary : Annotated[str," Return a brief summary of the review"]
    # sentiment : Annotated[str," Return the sentiment of the review as 'positive', 'negative', or 'neutral'"]
    # /if you want literally just the sentiment as Positive, Negative or Neutral then you can use Literal type instead of Annotated
    sentiment : Literal['positive','negative','neutral']


structured_model = model.with_structured_output(Review2)

result = structured_model.invoke(""" The hardware is great, but the software feels bloated. There are too many pre_installed apps that i cant remove.
             Also the UI looks outdated compared to other brands . Hoping for a software update to fix this. """)

print(result)