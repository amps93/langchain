"""
pipenv install -q langchain_google_genai
"""
from dotenv import load_dotenv
import numpy as np
from numpy.linalg import norm

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)

print(len(embeddings), len(embeddings[0]))

embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')


def cos_sim(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))


for embedding in embeddings:
    print(cos_sim(embedding, embedded_query))

