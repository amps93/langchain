"""
구분	| 내용
지시 | 언어 모델에게 어떤 작업을 수행하도록 요청하는 구체적인 지시.
예시 | 요청된 작업을 수행하는 방법에 대한 하나 이상의 예시.
맥락 | 특정 작업을 수행하기 위한 추가적인 맥락
질문 | 어떤 답변을 요구하는 구체적인 질문.

예시: 제품 리뷰 요약
지시: "아래 제공된 제품 리뷰를 요약해주세요."
예시: "예를 들어, '이 제품은 매우 사용하기 편리하며 배터리 수명이 길다'라는 리뷰는 '사용 편리성과 긴 배터리 수명이 특징'으로 요약할 수 있습니다."
맥락: "리뷰는 스마트워치에 대한 것이며, 사용자 경험에 초점을 맞추고 있습니다."
질문: "이 리뷰를 바탕으로 스마트워치의 주요 장점을 두세 문장으로 요약해주세요."
"""

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 'name'과 'age'라는 두 개의 변수를 사용하는 프롬프트 템플릿을 정의
template_text = "안녕하세요, 제 이름은 {name}이고, 나이는 {age}살입니다."

# PromptTemplate 인스턴스를 생성
prompt_template = PromptTemplate.from_template(template_text)

# 템플릿에 값을 채워서 프롬프트를 완성
filled_prompt = prompt_template.format(name="홍길동", age=30)

print(filled_prompt)

"""
프롬프트 템플릿 간의 결합
"""
# 문자열 템플릿 결합 (PromptTemplate + PromptTemplate + 문자열)
combined_prompt = (
              prompt_template
              + PromptTemplate.from_template("\n\n아버지를 아버지라 부를 수 없습니다.")
              + "\n\n{language}로 번역해주세요."
)

print(combined_prompt)
print(combined_prompt.format(name="홍길동", age=30, language="영어"))

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo")
chain = combined_prompt | llm | StrOutputParser()
print(chain.invoke({"age": 30, "language": "영어", "name": "홍길동"}))
