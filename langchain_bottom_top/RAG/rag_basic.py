"""
RAG(Retrieval-Augmented Generation) 파이프라인은 기존의 언어 모델에 검색 기능을 추가하여,
주어진 질문이나 문제에 대해 더 정확하고 풍부한 정보를 기반으로 답변을 생성할 수 있게 해줍니다.
이 파이프라인은 크게 데이터 로드, 텍스트 분할, 인덱싱, 검색, 생성의 다섯 단계로 구성됩니다.
"""
from dotenv import load_dotenv

load_dotenv()

# 1. 데이터 로드: RAG에 사용할 데이터를 불러오는 단계
# Data Loader - 웹페이지 데이터 가져오기
from langchain_community.document_loaders import WebBaseLoader

# 위키피디아 정책과 지침
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)

# 웹페이지 텍스트 -> Documents
docs = loader.load()

print(len(docs))
print(len(docs[0].page_content))
print(docs[0].page_content[5000:6000])
print('-------------------------------------------------------------------------')

# 2. 텍스트 분할: 불러온 데이터를 작은 크기의 단위(chunk)로 분할하는 과정
# Text Split (Documents -> small chunks: Documents)
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(len(splits))
print(splits[10])

# page_content 속성
print(splits[10].page_content)

# metadata 속성
print(splits[10].metadata)
print('-------------------------------------------------------------------------')

# 3. 인덱싱: 분할된 텍스트를 검색 가능한 형태로 만드는 단계
# Indexing (Texts -> Embedding -> Store)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# OpenAI의 임베딩 모델을 사용하여 텍스트를 벡터로 변환하고, 이를 Chroma 벡터저장소에 저장
vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

# vectorstore.similarity_search: 주어진 쿼리 문자열에 대해 저장된 문서들 중에서 가장 유사한 문서들을 찾아냄
docs = vectorstore.similarity_search("격하 과정에 대해서 설명해주세요.")
print(len(docs))
print(docs[0].page_content)
print('-------------------------------------------------------------------------')

# 4. 검색(Retrieval)
# 사용자의 질문이나 주어진 컨텍스트에 가장 관련된 정보를 찾아내는 과정
# 사용자의 입력을 바탕으로 쿼리를 생성하고, 인덱싱된 데이터에서 가장 관련성 높은 정보를 검색
# LangChain의 retriever 메소드를 사용

# 5. 생성(Generation): 검색된 정보를 바탕으로 사용자의 질문에 답변을 생성
# LLM 모델에 검색 결과와 함께 사용자의 입력을 전달
# 모델은 사전 학습된 지식과 검색 결과를 결합하여 주어진 질문에 가장 적절한 답변을 생성
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

# LLM
model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Rretriever
# Chroma 벡터 스토어를 검색기로 사용하여 사용자의 질문과 관련된 문서를 검색
retriever = vectorstore.as_retriever()


# Combine Documents
# 검색된 문서들을 하나의 문자열로 반환
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)


# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Chain 실행
print(rag_chain.invoke("격하 과정에 대해서 설명해주세요."))

