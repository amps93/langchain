from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
"""
멀티 체인(Multi-Chain): 여러 개의 체인을 연결하거나 복합적으로 작용하는 것
"""
prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
prompt2 = ChatPromptTemplate.from_template(
    "explain {english_word} using oxford dictionary to me in Korean."
)

llm = ChatOpenAI(model="gpt-3.5-turbo")

chain1 = prompt1 | llm | StrOutputParser()

print(chain1.invoke({"korean_word": "미래"}))


chain2 = (
    {"english_word": chain1}
    | prompt2
    | llm
    | StrOutputParser()
)

print(chain2.invoke({"korean_word": "미래"}))
