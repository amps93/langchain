from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents(
    [
        '안녕하세요!',
        '어! 오랜만이에요',
        '이름이 어떻게 되세요?',
        '날씨가 추워요',
        'Hello LLM!'
    ]
)
print('임베딩된 문서의 개수:', len(embeddings), '임베딩 차원 수:', len(embeddings[0]))
# 1536차원 중에서 앞에서 20차원에 해당하는 원소 출력
print('변환된 임베딩 벡터를 출력\n', embeddings[0][:20])

embedded_query = embeddings_model.embed_query('첫인사를 하고 이름을 물어봤나요?')
print('생성된 임베딩 벡터의 처음 5개 원소를 슬라이싱하여 반환\n', embedded_query[:5])

# 코사인 유사도
import numpy as np
from numpy.linalg import norm


def cos_sim(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))


print('임베딩된 모든 문서와 쿼리 사이의 코사인 유사도를 출력')
for embedding in embeddings:
    print(cos_sim(embedding, embedded_query))
