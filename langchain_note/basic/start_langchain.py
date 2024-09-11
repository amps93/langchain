from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


# LangSmith 추적을 설정합니다. https://smith.langchain.com
# .env 파일에 LANGCHAIN_API_KEY를 입력합니다.
# !pip install -qU langchain-teddynote
# from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
# logging.langsmith("CH01-Basic")

# 객체 생성
"""
temperature: 사용할 샘플링 온도는 0과 2 사이에서 선택합니다. 0.8과 같은 높은 값은 출력을 더 무작위하게 만들고, 
0.2와 같은 낮은 값은 출력을 더 집중되고 결정론적으로 만듭니다.
max_tokens: 채팅 완성에서 생성할 토큰의 최대 개수입니다.
model_name: 적용 가능한 모델 리스트 - gpt-3.5-turbo - gpt-4-turbo - gpt-4o
"""
llm = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-3.5-turbo",  # 모델명
)

# 질의내용
question = "대한민국의 수도는 어디인가요?"

# 질의
response = llm.invoke(question)

print(response)
print(response.content)
print(response.response_metadata)
