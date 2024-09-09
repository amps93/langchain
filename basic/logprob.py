from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# LogProb
"""
주어진 텍스트에 대한 모델의 토큰 확률의 로그 값 을 의미합니다. 토큰이란 문장을 구성하는 개별 단어나 문자 등의 요소를 의미하고, 
확률은 모델이 그 토큰을 예측할 확률을 나타냅니다.
"""
# 객체 생성
llm_with_logprob = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    max_tokens=2048,  # 최대 토큰수
    model_name="gpt-3.5-turbo",  # 모델명
).bind(logprobs=True)

# 질의내용
question = "대한민국의 수도는 어디인가요?"

# 질의
response = llm_with_logprob.invoke(question)

# 결과 출력
print(response.response_metadata)
