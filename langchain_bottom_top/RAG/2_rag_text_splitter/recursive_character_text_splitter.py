from langchain_community.document_loaders import TextLoader

loader = TextLoader('history.txt', encoding='utf8')
data = loader.load()
"""
RecursiveCharacterTextSplitter
* 텍스트를 재귀적으로 분할하여 의미적으로 관련 있는 텍스트 조각들이 같이 있도록 하는 목적으로 설계됨
* 이 과정에서 문자 리스트(['\n\n', '\n', ' ', ''])의 문자를 순서대로 사용하여 텍스트를 분할하며, 
분할된 청크들이 설정된 chunk_size보다 작아질 때까지 이 과정을 반복

parameter
* chunk_size: 분할된 청크들이 chunk_size보다 작아질 때까지 반복
* chunk_overlap: 분할된 텍스트 조각들 사이에서 중복으로 포함될 문자 수를 정의
* length_function=len: 분할의 기준이 되는 길이를 측정하는 함수로 문자열의 길이를 반환하는 len 함수를 사용한다는 의미
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

texts = text_splitter.split_text(data[0].page_content)

print(len(texts))
# RecursiveCharacterTextSplitter의 작동 방식에 따라, 이 길이들은 대체로 chunk_size에 가깝거나 그보다 작게 분할
print(len(texts[0]), len(texts[1]), len(texts[2]))
# CharacterTextSplitter 클래스와 다르게 문장이 온전하게 유지된 채로 나누어짐
print(texts[0][-20:])
