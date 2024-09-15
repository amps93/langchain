from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader('./')
data = loader.load()

print(len(data))
print(data[0])
print(data[-3])
