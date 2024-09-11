"""
ChatPromptTemplate: 대화형 상황에서 여러 메시지 입력을 기반으로 단일 메시지 응답을 생성하는 데 사용
대화형 모델이나 챗봇 개발에 주로 사용
입력은 여러 메시지를 원소로 갖는 리스트로 구성되며, 각 메시지는 역할(role)과 내용(content)으로 구성

* 메시지 유형
SystemMessage: 시스템의 기능을 설명합니다.
HumanMessage: 사용자의 질문을 나타냅니다.
AIMessage: AI 모델의 응답을 제공합니다.
FunctionMessage: 특정 함수 호출의 결과를 나타냅니다.
ToolMessage: 도구 호출의 결과를 나타냅니다.
"""
from dotenv import load_dotenv

load_dotenv()

"""
튜플 형태의 메시지 목록으로 프롬프트 생성 (type, content)
"""
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
    ("user", "{user_input}"),
])

messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
print(messages)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-3.5-turbo')
chain = chat_prompt | llm | StrOutputParser()

print(chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"}))

"""
MessagePromptTemplate 활용
"""
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ]
)

messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
print(messages)

chain = chat_prompt | llm | StrOutputParser()

print(chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"}))
