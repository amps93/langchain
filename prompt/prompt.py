from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(model='gpt-3.5-turbo')

"""
방법 1. from_template() 메소드를 사용하여 PromptTemplate 객체 생성
치환될 변수를 { 변수 } 로 묶어서 템플릿을 정의
"""
# template 정의. {country}는 변수로, 이후에 값이 들어갈 자리를 의미
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt = PromptTemplate.from_template(template)
print(prompt)
print(prompt.format(country='대한민국'))  # prompt 생성. format 메소드를 이용하여 변수에 값을 넣어줌

chain = prompt | llm
print(chain.invoke('대한민국').content)

"""
방법 2. PromptTemplate 객체 생성과 동시에 prompt 생성
추가 유효성 검사를 위해 input_variables 를 명시적으로 지정
이러한 변수는 인스턴스화 중에 템플릿 문자열에 있는 변수와 비교하여 불일치하는 경우 예외를 발생시킴
"""
# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt = PromptTemplate(
    template=template,
    input_variables=["country"],
)
print(prompt)
print(prompt.format(country="대한민국"))

# template 정의
template = "{country1}과 {country2}의 수도는 각각 어디인가요?"

# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt = PromptTemplate(
    template=template,
    input_variables=["country1"],
    partial_variables={
        "country2": "미국"  # dictionary 형태로 partial_variables를 전달
    },
)
print(prompt)
# partial_variables: 부분 변수 채움
# partial을 사용하는 일반적인 용도는 함수를 부분적으로 사용하는 것입니다. 사용 사례는 항상 공통된 방식으로 가져오고 싶은 변수가 있는 경우입니다.
prompt_partial = prompt.partial(country2="캐나다")
print(prompt_partial)

chain = prompt | llm
print(chain.invoke("대한민국").content)

chain = prompt_partial | llm
print(chain.invoke("대한민국").content)
print(chain.invoke({"country1": "대한민국", "country2": "호주"}).content)

"""
3. 파일로부터 template 읽어오기
"""
from langchain_core.prompts import load_prompt

prompt = load_prompt("fruit_color.yaml", encoding="UTF8")
print(prompt)
print(prompt.format(fruit="사과"))

prompt2 = load_prompt("capital.yaml", encoding="UTF8")
print(prompt2.format(country="대한민국"))

"""
ChatPromptTemplate: 대화목록을 프롬프트로 주입하고자 할 때 활용할 수 있습니다.
메시지는 튜플(tuple) 형식으로 구성하며, (role, message) 로 구성하여 리스트로 생성할 수 있습니다.

role
 - "system": 시스템 설정 메시지 입니다. 주로 전역설정과 관련된 프롬프트입니다.
 - "human" : 사용자 입력 메시지 입니다. - "ai": AI 의 답변 메시지입니다.
"""
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template("{country}의 수도는 어디인가요?")
print(chat_prompt)
print(chat_prompt.format(country="대한민국"))

chat_template = ChatPromptTemplate.from_messages(
    [
        # role, message
        ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
        ("human", "반가워요!"),
        ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
        ("human", "{user_input}"),
    ]
)

# 챗 message 를 생성합니다.
messages = chat_template.format_messages(
    name="테디", user_input="당신의 이름은 무엇입니까?"
)
print(messages)
# 생성한 메시지를 바로 주입하여 결과 반환
print(llm.invoke(messages).content)

# 체인 생성
chain = chat_template | llm
print(chain.invoke({"name": "Teddy", "user_input": "당신의 이름은 무엇입니까?"}).content)

"""
MessagePlaceholder
LangChain은 포맷하는 동안 렌더링할 메시지를 완전히 제어할 수 있는 MessagePlaceholder 를 제공
메시지 프롬프트 템플릿에 어떤 역할을 사용해야 할지 확실하지 않거나 서식 지정 중에 메시지 목록을 삽입하려는 경우 유용할 수 있습니다.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 요약 전문 AI 어시스턴트입니다. 당신의 임무는 주요 키워드로 대화를 요약하는 것입니다.",
        ),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "지금까지의 대화를 {word_count} 단어로 요약합니다."),
    ]
)
# conversation 대화목록을 나중에 추가하고자 할 때 MessagesPlaceholder를 사용할 수 있습니다.
formatted_chat_prompt = chat_prompt.format(
    word_count=5,
    conversation=[
        ("human", "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ],
)
print(formatted_chat_prompt)

# chain 생성
chain = chat_prompt | llm | StrOutputParser()
# chain 실행 및 결과확인
print(chain.invoke(
    {
        "word_count": 5,
        "conversation": [
            (
                "human",
                "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다.",
            ),
            ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
        ],
    }
)
)
