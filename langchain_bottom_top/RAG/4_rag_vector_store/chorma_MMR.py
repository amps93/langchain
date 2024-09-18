"""
MMR (Maximum Marginal Relevance, MMR)
최대 한계 관련성(Maximum Marginal Relevance, MMR) 검색 방식은 유사성과 다양성의 균형을 맞추어 검색 결과의 품질을 향상시키는 알고리즘입니다.
이 방식은 검색 쿼리에 대한 문서들의 관련성을 최대화하는 동시에, 검색된 문서들 사이의 중복성을 최소화하여, 사용자에게 다양하고 풍부한 정보를 제공하는 것을 목표로 합니다.

쿼리에 대한 각 문서의 유사성 점수와 이미 선택된 문서들과의 다양성(또는 차별성) 점수를 조합하여, 각 문서의 최종 점수를 계산
계산식 - https://wikidocs.net/231585

parameter
* query: 사용자로부터 입력받은 검색 쿼리입니다.
* k: 최종적으로 선택할 문서의 수입니다. 이 매개변수는 반환할 문서의 총 개수를 결정합니다.
* fetch_k: MMR 알고리즘을 수행할 때 고려할 상위 문서의 수 입니다. 이는 초기 후보 문서 집합의 크기를 의미하며, 이 중에서 MMR에 의해 최종 문서가 k개 만큼 선택됩니다.
* lambda_mult: 쿼리와의 유사성과 선택된 문서 간의 다양성 사이의 균형을 조절합니다. (lambda=1)은 유사성만 고려하며, (lambda=0)은 다양성만을 최대화합니다.
"""
# Load data -> Text split

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyMuPDFLoader('카카오뱅크 2023 지속가능경영보고서.pdf')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

documents = text_splitter.split_documents(data)
print(len(documents))

# Embedding -> Upload to Vectorstore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()
db2 = Chroma.from_documents(
    documents,
    embeddings_model,
    collection_name='esg',
    persist_directory='./db/chromadb',
    collection_metadata={'hnsw:space': 'cosine'},  # l2 is the default
)

print(db2)

"""
1. 일반적인 유사도 기반 검색
* 쿼리 '카카오뱅크의 환경목표와 세부추진내용을 알려줘?'를 사용하여 db2에서 유사성 검색을 수행합니다.
* len(docs)는 반환된 문서의 총 수를 출력합니다.
* docs[0].page_content는 검색 결과 중 가장 유사한 문서의 내용을 출력합니다.
"""
query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘?'
docs = db2.similarity_search(query)
print(len(docs))
print(docs[0].page_content)
print('-----------------------------------------')

# 검색 결과 중 가장 유사도가 낮은(또는 마지막에 위치한) 문서의 내용을 출력
print(docs[-1].page_content)
print('-----------------------------------------')

"""
2. MMR 검색
* 동일한 쿼리를 사용하여 MMR 검색을 수행합니다. 여기서는 k=4와 fetch_k=10을 설정하여, 상위 10개의 유사한 문서 중에서 서로 다른 정보를 제공하는 4개의 문서를 선택합니다.
* len(mmr_docs)는 MMR 검색으로 선택된 문서의 총 수, 여기서는 4개를 출력합니다.
* mmr_docs[0].page_content는 MMR 검색 결과 중 가장 높은 순위의 문서의 내용을 출력합니다.
"""
mmr_docs = db2.max_marginal_relevance_search(query, k=4, fetch_k=10)
print(len(mmr_docs))
print('-----------------------------------------')
print(mmr_docs[0].page_content)
print('-----------------------------------------')
# MMR 검색 결과 중 가장 낮은 순위(이 경우 4번째)의 문서의 내용을 출력
print(mmr_docs[-1].page_content)
