from langchain_community.document_loaders import TextLoader

loader = TextLoader('history.txt', encoding='utf8')
data = loader.load()
"""
CharacterTextSplitter(): 주어진 텍스트를 문자 단위로 분할하는 데 사용
parameter
* separator: 분할된 각 청크를 구분할 때 기준이 되는 문자열입니다. 여기서는 빈 문자열('')을 사용하므로, 각 글자를 기준으로 분할합니다.
* chunk_size: 각 청크의 최대 길이입니다. 여기서는 500으로 설정되어 있으므로, 최대 500자까지의 텍스트가 하나의 청크에 포함됩니다.
* chunk_overlap: 인접한 청크 사이에 중복으로 포함될 문자의 수입니다. 여기서는 100으로 설정되어 있으므로, 각 청크들은 연결 부분에서 100자가 중복됩니다.
* length_function: 청크의 길이를 계산하는 함수입니다. 여기서는 len 함수가 사용되었으므로, 문자열의 길이를 기반으로 청크의 길이를 계산합니다.
"""
"""
1. 문서를 개별 문자를 단위로 나누기 (separator="")
"""
# 각 문자를 구분하여 분할
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator='',
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

texts = text_splitter.split_text(data[0].page_content)

# 분할된 텍스트 조각 개수
print(len(texts))
# 분할된 텍스트 조각 중에서 첫번째 청크의 길이를 확인
print(len(texts[0]))
print(texts[0][-20:])

"""
2. 문서를 특정 문자열을 기준으로 나누기 (separator="문자열")
"""
# 줄바꿈 문자를 기준으로 분할
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
texts = text_splitter.split_text(data[0].page_content)
print(len(texts))
print(len(texts[0]), len(texts[1]), len(texts[2]))
print(texts[0][-20:])

