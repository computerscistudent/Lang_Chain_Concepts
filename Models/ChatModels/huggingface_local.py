from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'  # Set the Hugging Face cache directory

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation",
    pipeline_kwargs = {
        "max_new_tokens": 100,
        "temperature": 0.1,
        "do_sample": True,
    }
)

model = ChatHuggingFace(llm = llm)
result = model.invoke("Give exactly 5 movie names. No explanation. No extra text. Only a numbered list.",
                    stop=["\n\n", "See", "Haha", "#"])

print("\n🎬 TinyLlama Output:")
print(result.content)