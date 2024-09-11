"""
표준 인터페이스
stream: 응답의 청크를 스트리밍합니다.
invoke: 입력에 대해 체인을 호출합니다.
batch: 입력 목록에 대해 체인을 호출합니다.

비동기 메소드 - 비동기는 동시에 일어나지 않는다를 의미한다. 요청과 결과가 동시에 일어나지 않을 거라는 약속이다.
하나의 요청에 따른 응답을 즉시 처리하지 않아도, 그 대기 시간동안 또 다른 요청에 대해 처리 가능한 방식이다.
여러 개의 요청을 동시에 처리할 수 있는 장점이 있지만 동기 방식보다 속도가 떨어질 수도 있다.
astream: 비동기적으로 응답의 청크를 스트리밍합니다.
ainvoke: 비동기적으로 입력에 대해 체인을 호출합니다.
abatch: 비동기적으로 입력 목록에 대해 체인을 호출합니다.
astream_log: 최종 응답뿐만 아니라 발생하는 중간 단계를 스트리밍합니다.
"""

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_teddynote.messages import stream_response

# ChatOpenAI 모델을 인스턴스화합니다.
model = ChatOpenAI(model="gpt-3.5-turbo")
# model = Ollama(model='eeve')

# 주어진 토픽에 대한 농담을 요청하는 프롬프트 템플릿을 생성합니다.
prompt = PromptTemplate.from_template("{topic} 에 대하여 3문장으로 설명해줘.")

# 프롬프트와 모델을 연결하여 대화 체인을 생성합니다.
chain = prompt | model | StrOutputParser()

"""
stream: 실시간 출력
"""
# chain.stream 메서드를 사용하여 '멀티모달' 토픽에 대한 스트림을 생성하고 반복합니다.
print("==========stream==========")
for token in chain.stream({"topic": "멀티모달"}):
    # 스트림에서 받은 데이터의 내용을 출력합니다. 줄바꿈 없이 이어서 출력하고, 버퍼를 즉시 비웁니다.
    print(token, end="", flush=True)
print()

"""
invoke: 호출
"""
print("==========invoke==========")
print(chain.invoke({"topic": "Chatgpt"}))

"""
batch: 배치(단위 실행)
여러 개의 딕셔너리를 포함하는 리스트를 인자로 받아, 각 딕셔너리에 있는 topic 키의 값을 사용하여 일괄 처리를 수행
"""
print("==========batch==========")
# 주어진 토픽 리스트를 batch 처리하는 함수 호출
print(chain.batch([{"topic": "ChatGPT"}, {"topic": "Instagram"}]))

print(
    chain.batch(
        [
            {"topic": "ChatGPT"},
            {"topic": "Instagram"},
            {"topic": "멀티모달"},
            {"topic": "프로그래밍"},
            {"topic": "머신러닝"},
        ],
        config={"max_concurrency": 3},  # 매개변수를 사용하여 동시 요청 수를 설정
    )
)

"""
책에 나와있는 async 사용법은 jupyter에서만 가능
파이참 환경에서 사용하려면 파이썬 비동기 프로그래밍 공부(async, await, asyncio) 필요
langchain 비동기 실습은 LECL_interface.ipynb 파일에서 진행
"""

"""
Parallel: 병렬성
여러 작업을 병렬로 실행 - RunnableParallel 사용
"""
print("==========parallel==========")
from langchain_core.runnables import RunnableParallel

# {country} 의 수도를 물어보는 체인을 생성합니다.
chain1 = (
    PromptTemplate.from_template("{country} 의 수도는 어디야?")
    | model
    | StrOutputParser()
)

# {country} 의 면적을 물어보는 체인을 생성합니다.
chain2 = (
    PromptTemplate.from_template("{country} 의 면적은 얼마야?")
    | model
    | StrOutputParser()
)

# 위의 2개 체인을 동시에 생성하는 병렬 실행 체인을 생성합니다.
combined = RunnableParallel(capital=chain1, area=chain2)

# chain1 를 실행합니다.
print(chain1.invoke({"country": "대한민국"}))
# chain2 를 실행합니다.
print(chain2.invoke({"country": "미국"}))
# 병렬 실행 체인을 실행합니다.
print(combined.invoke({"country": "대한민국"}))

"""
배치에서의 병렬 처리
"""
# 배치 처리를 수행합니다.
print(chain1.batch([{"country": "대한민국"}, {"country": "미국"}]))
# 배치 처리를 수행합니다.
print(chain2.batch([{"country": "대한민국"}, {"country": "미국"}]))
# 주어진 데이터를 배치로 처리합니다.
print(combined.batch([{"country": "대한민국"}, {"country": "미국"}]))
