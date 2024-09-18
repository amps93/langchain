"""
멀티 쿼리 검색도구(MultiQueryRetriever)는 벡터스토어 검색도구(Vector Store Retriever)의 한계를 극복하기 위해 고안된 방법입니다.
사용자가 입력한 쿼리의 의미를 다각도로 포착하여 검색 효율성을 높이고, LLM을 활용하여 사용자에게 보다 관련성 높고 정확한 정보를 제공하는 것을 목표로 합니다.

MultiQueryRetriever 클래스를 사용하여 여러 쿼리에 기반한 문서 검색 과정을 설정하고 실행하는 방법

1. MultiQueryRetriever 설정: from_llm 메서드를 통해, 기존 벡터저장소 검색도구(vectorstore.as_retriever())와 LLM 모델(llm)을
결합하여 MultiQueryRetriever 인스턴스를 생성합니다. 이때 LLM은 다양한 관점의 쿼리를 생성하는 데 사용됩니다.

2. 로깅 설정: 로깅을 설정하여 MultiQueryRetriever에 의해 생성되고 실행되는 쿼리들에 대한 정보를 로그로 기록하고 확인할 수 있습니다.
검색 과정에서 어떤 쿼리들이 생성되고 사용되었는지 이해하는 데 도움이 됩니다.

3. 문서 검색 실행: retriever_from_llm.invoke 메서드를 사용하여 주어진 사용자 쿼리(question)에 대해 멀티 쿼리 기반의 문서 검색을 실행합니다.
생성된 모든 쿼리에 대해 문서를 검색하고, 중복을 제거하여 고유한 문서들만을 결과로 반환합니다.

4. 결과 확인: 검색을 통해 반환된 고유 문서들의 수를 확인합니다. 멀티 쿼리 접근 방식을 통해 얼마나 많은 관련 문서가 검색되었는지를 나타냅니다.
"""
from dotenv import load_dotenv
load_dotenv()

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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

# 멀티 쿼리 생성
question = '카카오뱅크의 최근 영업실적을 알려줘.'

llm = ChatOpenAI(
    model='gpt-3.5-turbo-0125',
    temperature=0,
    max_tokens=500,
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

unique_docs = retriever_from_llm.invoke(question)
print(len(unique_docs))
print(unique_docs[1])

"""
MultiQueryRetriever(retriever_from_llm)를 활용하여 여러 쿼리를 생성하고 검색된 문서를 기반으로 사용자 질문에 답변하는 과정

1. 프롬프트 설정: 사용자의 질문에 대한 답변을 생성하기 위한 프롬프트 템플릿을 정의합니다. 
여기서 {context}는 검색된 문서의 내용을 나타내고, {question}은 사용자의 질문입니다.

2. 모델 초기화: ChatOpenAI 클래스를 사용하여 GPT-3.5-turbo 모델을 초기화합니다. 
temperature를 0으로 설정하여 일관된 답변을 생성합니다.

3. 문서 포맷팅 함수 정의: 검색된 문서들을 하나의 문자열로 포맷팅하는 함수를 정의합니다. 
이 함수는 각 문서 내용을 두 개의 줄바꿈으로 구분하여 결합합니다.

4. 체인 정의 및 실행: RunnablePassthrough를 사용하여 사용자의 질문을 그대로 전달합니다. 
이후 검색된 문서(context)와 사용자의 질문(question)을 prompt에 전달하고, 생성된 프롬프트를 GPT-3.5-turbo 모델에 입력으로 제공합니다. 
StrOutputParser를 사용하여 모델의 출력을 문자열로 파싱합니다.

5. 실행 및 응답 생성: chain.invoke 메서드를 호출하여 전체 프로세스를 실행합니다. 
입력으로 "카카오뱅크의 최근 영업실적을 요약해서 알려주세요."라는 질문을 사용합니다. 
이 과정을 통해 검색된 문서의 내용을 기반으로 한 답변이 생성됩니다.
"""
# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''
from langchain.schema.runnable import RunnablePassthrough


prompt = ChatPromptTemplate.from_template(template)

# Model
llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
)


def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


# Chain
chain = (
    {'context': retriever_from_llm | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run
response = chain.invoke('카카오뱅크의 최근 영업실적을 요약해서 알려주세요.')
print(response)
