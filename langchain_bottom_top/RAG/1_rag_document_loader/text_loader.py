from langchain_community.document_loaders import TextLoader

loader = TextLoader('history.txt', encoding='UTF-8')
data = loader.load()

print(type(data))
print(len(data))

print(data)