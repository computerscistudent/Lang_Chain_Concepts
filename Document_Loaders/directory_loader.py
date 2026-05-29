from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader # type:ignore
)

docs = loader.load()

print(type(docs))
print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)

docs2 = loader.lazy_load()
print(type(docs2))

for doc in docs2:
    print(doc.metadata)

