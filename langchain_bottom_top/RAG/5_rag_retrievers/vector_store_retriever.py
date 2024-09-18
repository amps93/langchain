"""
Vector Store Retriver
벡터스토어 검색도구(Vector Store Retriever)를 사용하면 대량의 텍스트 데이터에서 관련 정보를 효율적으로 검색할 수 있습니다.
다음 코드에서는 LangChain의 벡터스토어와 임베딩 모델을 사용하여 문서들의 임베딩을 생성하고,
그 후 저장된 임베딩들을 기반으로 검색 쿼리에 가장 관련 있는 문서들을 검색하는 방법을 설명합니다.
"""
from dotenv import load_dotenv
load_dotenv()
"""
1. 사전 준비 - 문서 로드 및 분할
* PyMuPDFLoader를 사용하여 PDF 파일('323410_카카오뱅크_2023.pdf')에서 텍스트 데이터를 로드합니다. 
이 클래스는 PyMuPDF 라이브러리를 사용하여 PDF 문서의 내용을 추출합니다.
* RecursiveCharacterTextSplitter를 사용하여 문서를 텍스트 조각으로 분할하는 인스턴스를 생성하고 
text_splitter.split_documents(data)를 호출하여 로드된 문서 객체를 여러 개의 청크로 분할합니다.
* documents 변수에는 모두 145개의 문서 조각으로 분할되어 저장됩니다.
"""
# Load data -> Text split
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

loader = PyMuPDFLoader('카카오뱅크 2023 지속가능경영보고서.pdf')
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

documents = text_splitter.split_documents(data)
print(len(documents))

"""
2. 사전 준비 - 문서 임베딩을 벡터스토어에 저장
HuggingFaceEmbeddings를 사용하여 한국어 임베딩 모델인 'jhgan/ko-sbert-nli'를 사용
임베딩을 정규화하도록 설정
FAISS 벡터스토어를 사용하여 문서의 임베딩을 저장한 후 코사인 유사도를 측정 기준으로 사용해 유사도 측정
"""
# 벡터스토어에 문서 임베딩을 저장
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

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
3. 단일 문서 검색
검색 쿼리를 정의한 후, as_retriever 메소드를 사용하여 벡터스토어에서 Retriever 객체를 생성합니다. 
search_kwargs에서 k: 1을 설정하여 가장 유사도가 높은 하나의 문서를 검색합니다.
"""
# 검색 쿼리
query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘'

# 가장 유사도가 높은 문장을 하나만 추출
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

docs = retriever.invoke(query)
print('-----------단일 문서 검색-----------')
print(len(docs))
print(docs[0])

"""
4. MMR(Maximal Marginal Relevance) 검색 (1)
다양성을 고려한 MMR 검색을 사용하여 상위 5개 문서를 검색
여기서 fetch_k: 50는 후보 집합으로 선택되는 문서의 수, k: 5는 최종적으로 반환되는 문서의 수
lambda_mult: 0.5 설정은 유사도와 다양성 사이에서 적용될 수준. 0.5를 사용하면 중립적으로 적용
"""
# MMR - 다양성 고려 (lambda_mult = 0.5)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

docs = retriever.invoke(query)
print('-----------MMR_1-----------')
print(len(docs))
print(docs[0])

"""
5. MMR(Maximal Marginal Relevance) 검색 (2)
MMR은 검색 결과의 관련성과 다양성을 균형있게 조정하는 방식 
lambda_mult 매개변수는 관련성과 다양성 사이의 균형을 조정
여기서 lambda_mult가 0.15로 설정되어 있으므로, 관련성보다 다양성을 더 우선하게 된다
"""
# MMR - 다양성 고려 (lambda_mult 작을수록 더 다양하게 추출)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'lambda_mult': 0.15}
)

docs = retriever.invoke(query)
print('-----------MMR_2-----------')
print(len(docs))
print(docs[-1])

"""
6. 유사도 점수 임계값 기반 검색
설정한 score_threshold 유사도 점수 이상인 문서만을 대상으로 추출
여기서 임계값은 0.3으로 설정되어 있음
이는 쿼리 문장과 최소한 0.3 이상의 유사도를 가진 문서만을 검색 결과로 반환하게 됨. 따라서 유사도가 높은 문서만 필터링하고 싶을 때 유용하다
"""
print('-----------유사도 점수 임계값 기반 검색-----------')
# Similarity score threshold (기준 스코어 이상인 문서를 대상으로 추출)
retriever = vectorstore.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.3}
)

docs = retriever.invoke(query)
print('-----------유사도 점수 임계값 기반 검색-----------')
print(len(docs))
print(docs[0])
"""
7. 메타데이터 필터링을 사용한 검색
메타데이터의 특정 필드에 대해서 기준(예: 'format': 'PDF 1.4')을 설정하고 조건을 충족하는 문서만을 필터링하여 검색
특정 형식이나 조건을 만족하는 문서를 검색할 때 유용하다
"""

# 문서 객체의 metadata를 이용한 필터링
retriever = vectorstore.as_retriever(
    search_kwargs={'filter': {'format': 'PDF 1.5'}}
)

docs = retriever.invoke(query)
print('-----------메타데이터 필터링을 사용한 검색-----------')
print(len(docs))
print(docs[0])
"""
8. Generation - 답변 생성
실제로 사용자 쿼리('카카오뱅크의 환경목표와 세부추진내용을 알려줘')에 대한 답변을 생성
벡터 저장소에서 문서를 검색한 다음, 이를 기반으로 ChatGPT 모델에 쿼리를 수행하는 end-to-end 프로세스를 구현
이 과정을 통해 사용자의 질문에 대한 의미적으로 관련이 있는 답변을 생성할 수 있음

1. 검색 (Retrieval): vectorstore.as_retriever를 사용하여 MMR(Maximal Marginal Relevance) 검색 방식으로 문서를 검색 
search_kwargs에 k: 5와 lambda_mult: 0.15를 설정하여 상위 5개의 관련성이 높으면서도 다양한 문서를 선택함

2. 프롬프트 생성 (Prompt): ChatPromptTemplate를 사용하여 쿼리에 대한 답변을 생성하기 위한 템플릿을 정의 
여기서 {context}는 검색된 문서의 내용이고, {question}은 사용자의 쿼리임

3. 모델 (Model): ChatOpenAI를 사용하여 OpenAI의 GPT 모델을 초기화합니다. 이 예에서는 'gpt-3.5-turbo-0125' 모델을 사용하며, 
temperature를 0으로 설정하여 결정론적인 응답을 생성하고, max_tokens를 500으로 설정하여 응답의 길이를 제한

4. 문서 포맷팅 (Formatting Docs): 검색된 문서(docs)를 포맷팅하는 format_docs 함수를 정의
이 함수는 각 문서의 page_content를 가져와 두 개의 문단 사이에 두 개의 줄바꿈을 삽입하여 문자열로 결합

5. 체인 실행 (Chain Execution): prompt | llm | StrOutputParser()를 사용하여 LLM 체인을 구성하고 실행 
프롬프트를 통해 정의된 쿼리를 모델에 전달하고, 모델의 응답을 문자열로 파싱

6. 실행 (Run): chain.invoke 메서드를 사용하여 체인을 실행합니다. context로는 포맷팅된 문서 내용이고, question은 사용자의 쿼리 
최종 응답은 response 변수에 저장됨
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Retrieval
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'lambda_mult': 0.15}
)

docs = retriever.invoke(query)

# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

# Model
llm = ChatOpenAI(
    model='gpt-3.5-turbo',
    temperature=0,
    max_tokens=500,
)


def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


# Chain
chain = prompt | llm | StrOutputParser()

# Run
response = chain.invoke({'context': (format_docs(docs)), 'question': query})
print('-----------Generation - 답변 생성-----------')
print(response)
