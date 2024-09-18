"""
Chroma는 임베딩 벡터를 저장하기 위한 오픈소스 소프트웨어로, LLM(대규모 언어 모델) 앱 구축을 용이하게 하는 핵심 기능을 수행합니다.

Chroma의 주요 특징은 다음과 같습니다:

* 임베딩 및 메타데이터 저장: 대규모의 임베딩 데이터와 이와 관련된 메타데이터를 효율적으로 저장할 수 있습니다.
* 문서 및 쿼리 임베딩: 텍스트 데이터를 벡터 공간에 매핑하여 임베딩을 생성할 수 있으며, 이를 통해 검색 작업이 가능합니다.
* 임베딩 검색: 사용자 쿼리에 기반하여 가장 관련성 높은 임베딩을 찾아내는 검색 기능을 제공합니다.
"""

"""
유사도 기반 검색
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# 1. 데이터 로드
# TextLoader 클래스를 사용해 history.txt 파일에서 텍스트 데이터를 로드합니다.
# 로드된 데이터는 data 변수에 저장됩니다.
loader = TextLoader('history.txt', encoding='utf8')
data = loader.load()

# 2. 텍스트 분할
# RecursiveCharacterTextSplitter를 사용하여 로드된 텍스트를 여러 개의 작은 조각으로 분할합니다.
# 분할된 텍스트 조각들은 texts 변수에 저장됩니다.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=50,
    encoding_name='cl100k_base'
)

texts = text_splitter.split_text(data[0].page_content)
print(texts[0])

# 3. 임베딩 모델 초기화
# OpenAIEmbeddings를 사용하여 OpenAI 임베딩 모델의 인스턴스를 생성합니다.
# 이 단계에서 Huggingface 또는 다른 임베딩 모델을 사용할 수 있습니다.
embeddings_model = OpenAIEmbeddings()

# 4. Chroma 벡터 저장소 생성
# Chroma.from_texts 메소드를 사용하여 분할된 텍스트들을 임베딩하고, 이 임베딩을 Chroma 벡터 저장소에 저장합니다.
# 저장소는 collection_name으로 구분되며, 여기서는 'history'라는 이름을 사용합니다.
# 저장된 데이터는 ./db/chromadb 디렉토리에 저장됩니다.
# collection_metadata에서 'hnsw:space': 'cosine'을 설정하여 유사도 계산에 코사인 유사도를 사용합니다.
db = Chroma.from_texts(
    texts,
    embeddings_model,
    collection_name='history',
    persist_directory='./db/chromadb',
    collection_metadata={'hnsw:space': 'cosine'},  # l2 is the default
)
print(db)

# 5. 유사도 기반 검색 수행
# query 변수에 검색 쿼리를 정의합니다.
# db.similarity_search 메소드를 사용하여 저장된 데이터 중에서 쿼리와 가장 유사한 문서를 찾습니다.
# 검색 결과를 docs 변수에 저장하고, 가장 유사한 문서의 내용은 docs[0].page_content를 통해 확인합니다.
# 이 과정을 통해, 주어진 쿼리('누가 한글을 창제했나요?')에 대해 가장 관련성 높은 텍스트 조각('...세종대왕이 한글을 창제하여...')을 찾아내고 있습니다.
query = '누가 한글을 창제했나요?'
docs = db.similarity_search(query)
print(docs[0].page_content)
