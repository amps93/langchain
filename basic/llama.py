from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

llm = Ollama(model="llama-ko")
# llm = Ollama(model='eeve')

print(llm.invoke("오늘 날씨에 대해 알려줘"))
