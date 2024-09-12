"""
LLM과 Chat Model
LLM은 주로 단일 요청에 대한 복잡한 출력을 생성하는 데 적합한 반면, Chat Model은 사용자와의 상호작용을 통한 연속적인 대화 관리에 더 적합합니다.
"""

"""
LLM
"""
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAI

llm = OpenAI()
print(llm.invoke('한국의 대표 관광지를 3군데 추천해주세요'))


"""
Chat model:  대화형 메시지를 입력으로 사용하고 대화형 메시지를 출력으로 반환하는 특수화된 LLM 모델

1. 대화형 입력과 출력: Chat Model은 대화의 연속성을 고려하여 입력된 메시지 리스트를 기반으로 적절한 응답 메시지를 생성합니다. 
챗봇, 가상 비서, 고객 지원 시스템 등 대화 기반 서비스에 어울립니다.
2. 다양한 모델 제공 업체와의 통합: 랭체인은 OpenAI, Cohere, Hugging Face 등 다양한 모델 제공 업체와의 통합을 지원합니다. 
이를 통해 개발자는 여러 소스의 Chat Models를 조합하여 활용할 수 있습니다.
3. 다양한 작동 모드 지원: 랭체인은 동기(sync), 비동기(async), 배치(batching), 스트리밍(streaming) 모드에서 모델을 사용할 수 있는 기능을 제공합니다. 
다양한 애플리케이션 요구사항과 트래픽 패턴에 따라 유연한 대응이 가능합니다.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()
# 시스템이 여행 전문가라는 정보와 사용자 입력을 포함하는 프롬프트
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 여행 전문가입니다."),
    ("user", "{user_input}"),
])

# 체인 정의
chain = chat_prompt | chat
print(chain.invoke({"user_input": "안녕하세요? 한국의 대표적인 관광지 3군데를 추천해주세요."}))

"""
LLM 모델 파라미터
1. Temperature: 생성된 텍스트의 다양성을 조정합니다. 값이 작으면 예측 가능하고 일관된 출력을 생성하는 반면, 값이 크면 다양하고 예측하기 어려운 출력을 생성합니다.
2. Max Tokens (최대 토큰 수): 생성할 최대 토큰 수를 지정합니다. 생성할 텍스트의 길이를 제한합니다.
3. Top P (Top Probability): 생성 과정에서 특정 확률 분포 내에서 상위 P% 토큰만을 고려하는 방식입니다. 이는 출력의 다양성을 조정하는 데 도움이 됩니다.
4. Frequency Penalty (빈도 패널티): 값이 클수록 이미 등장한 단어나 구절이 다시 등장할 확률을 감소시킵니다. 이를 통해 반복을 줄이고 텍스트의 다양성을 증가시킬 수 있습니다. (0~1)
5. Presence Penalty (존재 패널티): 텍스트 내에서 단어의 존재 유무에 따라 그 단어의 선택 확률을 조정합니다. 값이 클수록 아직 텍스트에 등장하지 않은 새로운 단어의 사용이 장려됩니다. (0~1)
6. Stop Sequences (정지 시퀀스): 특정 단어나 구절이 등장할 경우 생성을 멈추도록 설정합니다. 이는 출력을 특정 포인트에서 종료하고자 할 때 사용됩니다.
"""
# 모델 파라미터 설정
params = {
    "temperature": 0.7,         # 생성된 텍스트의 다양성 조정
    "max_tokens": 100,          # 생성할 최대 토큰 수
}

# 모델 인스턴스를 생성할 때 설정
model = ChatOpenAI(model="gpt-3.5-turbo", **params, frequency_penalty=0.5, presence_penalty=0.5, stop=["\n"])

# 모델 호출
question = "태양계에서 가장 큰 행성은 무엇인가요?"
response = model.invoke(input=question)

# 전체 응답 출력
print(response)

# 모델 파라미터 설정
params = {
    "temperature": 0.7,         # 생성된 텍스트의 다양성 조정
    "max_tokens": 10,          # 생성할 최대 토큰 수
}

# 모델 인스턴스를 호출할 때 전달
response = model.invoke(input=question, **params)

# 문자열 출력
print(response.content)

"""
binding - 모델 인스턴스에 파라미터를 추가로 제공
bind 메서드를 사용하는 방식의 장점은 특정 모델 설정을 기본값으로 사용하고자 할 때 유용하며, 특수한 상황에서 일부 파라미터를 다르게 적용하고 싶을 때 사용
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
    ("user", "{user_input}"),
])

model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=100)

messages = prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")

before_answer = model.invoke(messages)

# # binding 이전 출력
print('-----before binding-----')
print(before_answer)

# 모델 호출 시 추가적인 인수를 전달하기 위해 bind 메서드 사용 (응답의 최대 길이를 10 토큰으로 제한)
chain = prompt | model.bind(max_tokens=10)

after_answer = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})

# binding 이후 출력
print('-----after binding-----')
print(after_answer)
