from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from Youtube_chatbot.text_chunks import TextChunks
from Youtube_chatbot.transcript import Transcript
import os
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model= "text-embedding-3-small")

class VectorStore: 
    @staticmethod
    def create_and_store_vectors(documents:list[Document]):
        try:
            vector_store = FAISS.from_documents(documents=documents,embedding=embedding)
            return vector_store
        except Exception as e:
            print(f"Unexpected error ocurred -: {e}")
            return None
            

if __name__ == "__main__":
    video_id = "c64hqovEG-U"
    caption = Transcript().get_transcript(video_id=video_id)
    chunks = TextChunks.create_chunks(caption)
    vector_store = VectorStore.create_and_store_vectors(chunks)
    if vector_store:
        print(vector_store.index_to_docstore_id)