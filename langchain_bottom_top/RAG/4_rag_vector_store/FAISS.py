"""
FAISS(Facebook AI Similarity Search)는 Facebook AI Research에 의해 개발된 라이브러리로,
대규모 벡터 데이터셋에서 유사도 검색을 빠르고 효율적으로 수행할 수 있게 해줍니다.
FAISS는 특히 벡터의 압축된 표현을 사용하여 메모리 사용량을 최소화하면서도 검색 속도를 극대화하는 특징이 있습니다.

유사도 기준
* 'l2' (기본값): 유클리디안 거리를 기반으로 하는 유사도 측정 방법입니다. 두 벡터 간의 거리가 작을수록 더 유사하다고 평가합니다.
* 'ip' (내적): 내적 기반 유사도 측정 방법으로, 두 벡터의 방향성이 얼마나 유사한지를 평가합니다. 값이 클수록 더 유사하다고 판단합니다.
'* cosine': 코사인 유사도를 기반으로 하는 방법으로, 두 벡터의 각도가 작을수록 (즉, 방향이 더 유사할수록) 더 유사하다고 평가합니다.
내적과 유사하지만, 벡터의 크기에 영향을 받지 않습니다.
"""
"""
1. 유사도 기반 검색 (Similarity search)
pipenv install faiss-cpu sentence-transformers
"""
# 벡터스토어 db 인스턴스를 생성
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyMuPDFLoader('카카오뱅크 2023 지속가능경영보고서.pdf')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

documents = text_splitter.split_documents(data)
embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

vectorstore = FAISS.from_documents(documents,
                                   embedding=embeddings_model,
                                   distance_strategy=DistanceStrategy.COSINE
                                   )
print(vectorstore)

query = '카카오뱅크가 중대성 평가를 통해 도출한 6가지 중대 주제는 무엇인가?'
docs = vectorstore.similarity_search(query)
print(len(docs))
# print(docs[0].page_content)

"""
2. MMR (Maximum marginal relevance search)
"""
mmr_docs = vectorstore.max_marginal_relevance_search(query, k=4, fetch_k=10)
print(len(mmr_docs))
print(mmr_docs[0].page_content)

"""
3. FAISS DB를 로컬에 저장하기
vectorstore.save_local 메서드를 사용하여 로컬 파일 시스템에 벡터 스토어를 저장할 수 있습니다.
저장된 벡터 스토어를 다시 불러오기 위해서는 FAISS.load_local 클래스 메서드를 사용합니다.
"""
# save db
vectorstore.save_local('./db/faiss')
# load db
db3 = FAISS.load_local('./db/faiss', embeddings_model)
