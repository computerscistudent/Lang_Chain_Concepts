from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="layoffs.csv")

#docs = loader.load()

# print(type(docs))
# print(len(docs))

# print(docs[0])

# print(docs[0].page_content)
# print(docs[0].metadata["source"])


docs2 = loader.lazy_load()

for i, doc in enumerate(docs2):
    print(" index -:{}\n {} \n\n ".format(i , doc.page_content.strip()))