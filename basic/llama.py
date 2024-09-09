from langchain_community.llms import Ollama
from langchain import PromptTemplate

llm = Ollama(model='llama3.1')

print(llm.invoke("how's the weather today?"))