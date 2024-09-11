from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# prompt 와 llm 을 생성합니다.
prompt = PromptTemplate.from_template("{num} 의 10배는?")
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
parser = StrOutputParser()
# chain 을 생성합니다.
chain = prompt | llm

# chain 을 실행합니다.
print(chain.invoke({"num": 5}).content)

"""
RunnablePassthrough: 사용자가 입력한 내용을 어느 위치에 넣을지 지정해서 다음으로 넘겨줌
RunnableParallel: 병렬로 수행
RunnableLambda: 함수 사용
"""
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

# RunnablePassthrough
# RunnablePassthrough로 체인을 구성하는 예제
runnable_chain = {"num": RunnablePassthrough()} | prompt | llm

# dict 값이 RunnablePassthrough() 로 변경되었습니다. -> 위와 같이 key:value를 넣는 것이 아니라 value만 넣어도 됨
print(runnable_chain.invoke(10).content)
print(runnable_chain.invoke(20).content)

# 입력 키: num, 할당(assign) 키: new_num
chain_assign = (RunnablePassthrough.assign(new_num=lambda x: x["num"] * 3)).invoke({"num": 1})
print(chain_assign)

# RunnableParallel
# 프롬프트 두개 생성
prompt1 = PromptTemplate.from_template("{country}의 수도는 어디야?")
prompt2 = PromptTemplate.from_template("{country}의 인구수는 몇명이야?")

# 체인 두개 생성
runnable_parallel_chain1 = {'country': RunnablePassthrough()} | prompt1 | llm | parser
runnable_parallel_chain2 = {'country': RunnablePassthrough()} | prompt2 | llm | parser

# RunnableParallel로 체인 두개 묶기
map_chain = RunnableParallel(a=runnable_parallel_chain1, b=runnable_parallel_chain2)
print(map_chain.invoke('대한민국'))


# RunnableLambda

# 함수 생성
def combine_text(text):
    return text['a'] + " " + text['b']


# RunnableLambda로 combine_text 함수 적용
final_chain = (map_chain
               | {'info': RunnableLambda(combine_text)}
               | PromptTemplate.from_template('다음의 내용을 자연스럽게 교정해줘. 이모티콘을 적절한 곳에 추가해줘:\n{info}')
               | llm
               | parser
               )
print(final_chain.invoke('대한민국'))
