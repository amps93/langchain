LangChain

위키독스 및 유튜브 등 웹에서 공개된 랭체인 학습을 실습한 repo

1. [&lt;랭체인LangChain 노트&gt; - LangChain 한국어 튜토리얼🇰🇷 - WikiDocs](https://wikidocs.net/book/14314) - langchain_note

2. [랭체인(LangChain) 입문부터 응용까지 - WikiDocs](https://wikidocs.net/book/14473) - langchain_bottom_top

3. [모두의AI - YouTube](https://www.youtube.com/@AI-km1yn/videos) - almost_langchain

# LangChain
## 1. 랭체인 기본 흐름
프롬프트 -> 모델 -> 아웃풋 파서 -> invoke

```
prompt = PromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

print(chain.invoke({"input": "지구의 자전 주기는?"}))
```

## 2. 프롬프트
* PromptTemplate: 단순한 프롬프트 생성에 적합, 단순한 템플릿
```
prompt_template = PromptTemplate.from_template("안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다.")
```
* ChatPromptTemplate: 대화형 프롬프트를 위한 템플릿 생성에 적합, 복잡한 템플릿
``` 
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
    ("user", "{user_input}"),
])
```

## 3. 모델
* LLM: 일반 텍스트 입력을 받고 이에 대한 응답을 생성. 광범위한 언어 이해 및 텍스트 생성
```
llm = OpenAI()
result = llm("Explain what LangChain is.")
print(result)
```

* Chat Model: 메시지의 리스트를 입력으로 받고, 하나의 메시지를 반환. 챗봇

```
chat = ChatOpenAI()
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangChain?")
]
response = chat(messages)
print(response.content)
```
* 파라미터 
  * temperature: 생성되는 텍스트의 창의성을 조절. 값이 작으면 일관된 출력, 높으면 다양하고 예측 어려운 출력. 
  * max_tokens: 생성할 텍스트의 최대 토큰 수를 제한 
  * top_p: 출력의 확률 분포에서 상위 P% 토큰만을 고려. 출력의 다양성 조정에 도움이 됨 
  * frequency_penalty: 동일한 어휘가 반복되는 것을 줄이기 위한 패널티. 클수록 단어나 구절의 재등장 확률 감소. (0~1)
  * presence_penalty: 새로운 주제나 어휘가 더 자주 나오도록 유도하는 패널티. 클수록 새로운 단어 등장 확률 증가. (0~1)
  * Stop Sequences (정지 시퀀스): 특정 단어나 구절이 등장할 경우 생성을 멈추도록 설정. 출력을 특정 포인트에서 종료하고자 할 때 사용.
  * ```
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=100)
    ```
* bind: chain 단계에서 모델의 파라미터를 추가로 설정할 수 있음
  * ```
    model = ChatOpenAI(model="gpt-4o-mini", max_tokens=100)
    chain = prompt | model.bind(max_tokens=10)
    ```

## 4. 출력 파서
모델의 출력을 처리하고, 그 결과를 원하는 형식으로 변환. csv, json 등 다양한 방식으로 출력 가능
```
output_parser = StrOutputParser()  # JsonOutputParser, CommaSeparatedListOutputParser

chain = prompt | llm | output_parser
```

# RAG(Retrieval-Augmented Generation)
주어진 질문에 대해 언어 모델(LLM)과 검색 시스템을 결합하여 더 나은 답변을 생성하는 방식. 검색과 생성 두 단계로 구성됨
1. 검색 (Retrieval)
   1. 사용자가 입력한 질문에 맞춰 외부 데이터 소스(데이터베이스, 문서 저장소, API 등)에서 관련 정보를 검색 
   2. 이 단계에서 문서나 텍스트 조각들이 가져와지며, 이를 기반으로 답변을 생성 
   3. LangChain에서 이 작업은 Vector Store를 사용하여 수행. 여기서 벡터화된 문서를 검색하는 것이 일반적
2. 생성 (Generation)
   1. 검색된 정보를 바탕으로 언어 모델(LLM)이 최종 응답을 생성 
   2. 단순히 모델의 내재된 지식을 사용하여 생성하는 것이 아니라 검색된 자료를 기반으로 하여 답변의 정확성을 높임

## 1. Document Loader: 다양한 소스의 문서를 불러오고 처리
웹, 텍스트, csv, pdf 등 다양한 문서 로드 가능
1. 웹 문서 로드
```
import bs4
from langchain_community.document_loaders import WebBaseLoader

url1 = "https://blog.langchain.dev/week-of-1-22-24-langchain-release-notes/"

loader = WebBaseLoader(
    web_paths=(url1, url2),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("article-header", "article-content")
        )
    ),
)
docs = loader.load()
```
2. csv 로드
```
loader = CSVLoader(file_path='한국주택금융공사_주택금융관련_지수_20160101.csv', encoding='cp949')
data = loader.load()
```

### Text splitter
긴 문서를 작은 단위인 청크(chunk)로 나눔. CharacterTextSplitter, RecursiveCharacterTextSplitter 등이 있음
```
text_splitter = CharacterTextSplitter(
    separator = '',
    chunk_size = 500,
    chunk_overlap  = 100,
    length_function = len,
)
texts = text_splitter.split_text(data[0].page_content)
```
### 토큰 수 기준으로 텍스트 분할
모델이 처리할 수 있는 토큰 수에는 한계가 있음. 입력 데이터를 모델의 제한을 초과하지 않도록 적절히 분할하는 것이 중요.
tiktoken 라이브러리 사용해 토크나이저 기준으로 분할

글자 수 기준으로 분할할 때 tiktoken 토크나이저를 기준으로 글자 수를 계산하여 분할
```
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=200,
    encoding_name='cl100k_base'
)

docs = text_splitter.split_documents(data)
```

## Embedding
텍스트 데이터를 숫자로 이루어진 벡터로 변환하여 수치적 표현으로 만드는 과정. OpenAI, HuggingFace, Google 에서 임베딩 모델을 제공
```
embeddings_model = OpenAIEmbeddings()
embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)
```

## Vector Store
임베딩된 벡터들을 효율적으로 저장하고 검색할 수 있는 시스템 혹은 데이터베이스를 의미. Chroma, FAISS 등
```
embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)


vectorstore = FAISS.from_documents(documents,
                                   embedding = embeddings_model,
                                   distance_strategy = DistanceStrategy.COSINE
                                   )
```

## Retriever
벡터 저장소에서 문서를 검색하는 도구. 단일 Retriver, Multi Query Retriver, Contextual compression 등
```
# 검색 쿼리
query = '카카오뱅크의 환경목표와 세부추진내용을 알려줘'

# 가장 유사도가 높은 문장을 하나만 추출
retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

docs = retriever.invoke(query)
docs[0]
```