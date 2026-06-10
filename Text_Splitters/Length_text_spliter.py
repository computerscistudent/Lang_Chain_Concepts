from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

text = """
    Space is a breathtaking expanse of endless mystery, a silent vacuum stretching far beyond our imagination. It holds billions of galaxies, each teeming with billions of stars, planets, and cosmic wonders. 
    
    From the violent, swirling vortices of black holes to the serene beauty of colorful nebulae where new stars are born, the universe is both chaotic and magnificent. Our own home, Earth, is merely a tiny blue speck drifting in this vast cosmic ocean. As we peer deeper into the dark unknown with advanced telescopes, we continue to hunt for answers to humanity's oldest question: are we truly alone?
"""

loader = TextLoader(file_path="cricket.txt")

text2 = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=220,
    chunk_overlap=0,
    separator = "\n"
)

#rez = splitter.split_documents(text2)
rez = splitter.split_text(text2[0].page_content)

print(rez)
print(len(rez))


# the text2 is a list of documents, we need to access the page_content of the first document in the list to get the actual text content.
# print(text2) # str
# print(type(text2)) # list
# print(len(text2)) # 1
# print(text2[0].page_content)