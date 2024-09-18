"""
컨텍스트 압축 기법은 검색된 문서 중에서 쿼리와 관련된 정보만을 추출하여 반환하는 것을 목표로 합니다.
쿼리와 무관한 정보를 제거하는 방식으로 답변의 품질을 높이고 비용을 줄일 수 있습니다.
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

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

"""
기본 검색기 정의

1. 기본 검색기(Base Retriever) 설정
vectorstore.as_retriever 함수를 사용하여 기본 검색기를 설정합니다. 
여기서 search_type='mmr'와 search_kwargs={'k':7, 'fetch_k': 20}는 검색 방식을 설정합니다. 
mmr 검색 방식은 다양성을 고려한 검색 결과를 제공하여, 단순히 가장 관련성 높은 문서만 반환하는 대신 다양한 관점에서 관련된 문서들을 선택합니다.

2. 쿼리 처리 및 문서 검색
base_retriever.invoke(question) 함수를 사용하여 주어진 쿼리에 대한 관련 문서를 검색합니다.
이 함수는 쿼리와 관련성 높은 문서들을 반환합니다.

3. 결과 출력
print(len(docs))를 통해 검색된 문서의 수를 출력합니다.
"""
question = '카카오뱅크의 최근 영업실적을 알려줘.'

llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0,
    max_tokens=500,
)

base_retriever = vectorstore.as_retriever(
                                search_type='mmr',
                                search_kwargs={'k': 7, 'fetch_k': 20})

docs = base_retriever.invoke(question)
print(len(docs))
"""
문서 압축기의 구성과 작동 방식
문서 압축기는 기본 검색기로부터 얻은 문서들을 더욱 효율적으로 압축하여, 쿼리와 가장 관련이 깊은 내용만을 추려내는 것을 목표로 합니다. 
LLMChainExtractor와 ContextualCompressionRetriever 클래스를 사용합니다.

1. LLMChainExtractor 설정
LLMChainExtractor.from_llm(llm)를 사용하여 문서 압축기를 설정합니다. 언어 모델(llm)을 사용하여 문서 내용을 압축합니다.

2. ContextualCompressionRetriever 설정
ContextualCompressionRetriever 인스턴스를 생성할 때, base_compressor와 base_retriever를 인자로 제공합니다. 
base_compressor는 앞서 설정한 LLMChainExtractor 인스턴스이며, base_retriever는 기본 검색기 인스턴스입니다. 
이 두 구성 요소를 결합하여 검색된 문서들을 압축하는 과정을 처리합니다.

3. 압축된 문서 검색
compression_retriever.invoke(question) 함수를 사용하여 주어진 쿼리에 대한 압축된 문서들을 검색합니다. 
기본 검색기를 통해 얻은 문서들을 문서 압축기를 사용하여 내용을 압축하고, 쿼리와 가장 관련된 내용만을 추려냅니다.

4. 결과 출력
print(len(compressed_docs))를 통해 압축된 문서의 수를 출력합니다.
"""
# 문서 압축기를 연결하여 구성
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

compressed_docs = compression_retriever.invoke(question)
print(len(compressed_docs))
print(compressed_docs)
