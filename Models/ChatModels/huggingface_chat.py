from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os


load_dotenv()
repo_id="HuggingFaceH4/zephyr-7b-beta"

llm = HuggingFaceEndpoint(
    model=repo_id,
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    max_new_tokens=40,
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Give exactly 5 movie names. No explanation. No extra text. Only a numbered list.",
                    stop=["\n\n", "See", "Haha", "#"])

print(result.content)