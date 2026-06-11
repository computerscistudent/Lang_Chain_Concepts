from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from Youtube_chatbot.text_chunks import TextChunks
from Youtube_chatbot.transcript import Transcript
import os
from dotenv import load_dotenv
load_dotenv()

embedding = OpenAIEmbeddings(model= "text-embedding-3-small")

class VectorStoreRetriever: 
    @staticmethod
    def create_and_store_vectors(documents:list[Document]):
        try:
            vector_store = FAISS.from_documents(documents=documents,embedding=embedding)
            return vector_store
        except Exception as e:
            print(f"Unexpected error ocurred -: {e}")
            return None
    
    @staticmethod
    def build_retriever(vector_store):
        try:
            if vector_store:
                retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
                return retriever
        except Exception as e:
            print(f"Unexpected error ocurred -: {e}")
            return None
            

if __name__ == "__main__":
    video_id = "c64hqovEG-U"
    caption = Transcript().get_transcript(video_id=video_id)
    chunks = TextChunks.create_chunks(caption)
    vector_store = VectorStoreRetriever.create_and_store_vectors(chunks)
    if vector_store:
        print("Vector store got created successfully")
        #print(vector_store.index_to_docstore_id)

        retriever = VectorStoreRetriever.build_retriever(vector_store=vector_store)
        print("Retriever configured successfully!")

        if retriever:
            rez = retriever.invoke("Where are the sun's siblings")
            for i, doc in enumerate(rez):
                print(f"Document {i+1}: {doc.page_content}\n")
            context = "\n\n---\n\n".join(doc.page_content for doc in rez)
            #context = context.replace("\n", " ")
            print("\n\n")
            print(f"Context: {context}")
