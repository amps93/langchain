"""
LangChain Hub 에서 프롬프트를 받아서 실행하는 예제
아래 주소에서 LangChain Hub 프롬프트를 확인할 수 있습니다.
받아오는 방법은 프롬프트 repo 의 아이디 값을 가져 올 수 있고, commit id 를 붙여서 특정 버전에 대한 프롬프트를 받아올 수도 있습니다.
"""
from langchain import hub

# 가장 최신 버전의 프롬프트를 가져옵니다.
prompt = hub.pull("rlm/rag-prompt")
print(prompt)

# 특정 버전의 프롬프트를 가져오려면 버전 해시를 지정하세요
prompt = hub.pull("rlm/rag-prompt:50442af1")
print(prompt)

"""
Prompt Hub에 자신의 프롬프트 등록
"""
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "주어진 내용을 바탕으로 다음 문장을 요약하세요. 답변은 반드시 한글로 작성하세요\n\nCONTEXT: {context}\n\nSUMMARY:"
)
print(prompt)
from langchain import hub

# 프롬프트를 허브로부터 가져옵니다.
pulled_prompt = hub.pull("teddynote/simple-summary-korean")
# 프롬프트 내용 출력
print(pulled_prompt)
