from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter


loader = TextLoader('history.txt', encoding='utf8')
data = loader.load()
"""
대규모 언어 모델(LLM)을 사용할 때 모델이 처리할 수 있는 토큰 수에는 한계가 있습니다. 
입력 데이터를 모델의 제한을 초과하지 않도록 적절히 분할하는 것이 중요합니다. 
이때 LLM 모델에 적용되는 토크나이저를 기준으로 텍스트를 토큰으로 분할하고, 이 토큰들의 수를 기준으로 텍스트를 청크로 나누면 모델 입력 토큰 수를 조절할 수 있습니다.
"""
"""
CharacterTextSplitter.from_tiktoken_encoder
글자 수 기준으로 분할할 때 tiktoken 토크나이저를 기준으로 글자 수를 계산하여 분할

parameter
encoding_name='cl100k_base': 텍스트를 토큰으로 변환하는 인코딩 방식을 나타냄
"""
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

docs = text_splitter.split_documents(data)
print(len(docs))

print(len(docs[0].page_content))
print(docs[0])

print(len(docs[1].page_content))
print(docs[1])
