"""
출력 파서 (Output Parser): 모델의 출력을 처리하고, 그 결과를 원하는 형식으로 변환하는 역할을 함
모델에서 반환된 원시 텍스트를 분석하고, 특정 정보를 추출하거나, 출력을 특정 형식으로 재구성하는 데 사용됨
"""
"""
출력 파서 (Output Parser)의 주요 기능
출력 포맷 변경: 모델의 출력을 사용자가 원하는 형식으로 변환합니다. 예를 들어, JSON 형식으로 반환된 데이터를 테이블 형식으로 변환할 수 있습니다.
정보 추출: 원시 텍스트 출력에서 필요한 정보(예: 날짜, 이름, 위치 등)를 추출합니다. 이를 통해 복잡한 텍스트 데이터에서 구조화된 정보를 얻을 수 있습니다.
결과 정제: 모델 출력에서 불필요한 정보를 제거하거나, 응답을 더 명확하게 만드는 등의 후처리 작업을 수행합니다.
조건부 로직 적용: 출력 데이터를 기반으로 특정 조건에 따라 다른 처리를 수행합니다. 예를 들어, 모델의 응답에 따라 사용자에게 추가 질문을 하거나, 다른 모델을 호출할 수 있습니다.
"""
"""
출력 파서 (Output Parser)의 사용 사례
자연어 처리(NLP) 애플리케이션: 질문 답변 시스템에서 정확한 답변만을 추출하여 사용자에게 제공합니다.
데이터 분석: 대량의 텍스트 데이터에서 특정 패턴이나 통계 정보를 추출하여 분석 보고서를 생성합니다.
챗봇 개발: 대화형 모델의 출력을 분석하여 사용자의 의도를 파악하고, 적절한 대화 흐름을 유지합니다.
콘텐츠 생성: 생성된 콘텐츠에서 중요한 정보를 요약하거나, 특정 형식(예: 블로그 포스트, 뉴스 기사)에 맞게 콘텐츠를 재구성합니다.
"""
from dotenv import load_dotenv

load_dotenv()

"""
CSV parser
"""
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

print(format_instructions)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

chain = prompt | llm | output_parser

print(chain.invoke({"subject": "popular Korean cusine"}))

"""
Json parser
"""
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


# 자료구조 정의 (pydantic)
class CusineRecipe(BaseModel):
    name: str = Field(description="name of a cusine")
    recipe: str = Field(description="recipe to cook the cusine")


# 출력 파서 정의
output_parser = JsonOutputParser(pydantic_object=CusineRecipe)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

# prompt 구성
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)
print(prompt)

# 체인 생성
chain = prompt | llm | output_parser

print(chain.invoke({"query": "Let me know how to cook Bibimbap"}))
