"""
pipenv install sentence-transformers
pipenv install langchain-huggingface
HuggingFaceEmbeddings: HuggingFace의 트랜스포머 모델을 사용하여 문서 또는 문장을 임베딩하는 데 사용

parameter
* model_name='jhgan/ko-sroberta-nli': 사용할 모델을 지정. 여기서는 한국어 자연어 추론(Natural Language Inference, NLI)에 최적화된 ko-sroberta 모델을 사용
* model_kwargs={'device':'cpu'} : 모델이 CPU에서 실행되도록 설정. GPU를 사용할 수 있는 환경이라면 'cuda'로 설정할 수도 있음
* encode_kwargs={'normalize_embeddings':True} : 임베딩을 정규화하여 모든 벡터가 같은 범위의 값을 갖도록 함. 이는 유사도 계산 시 일관성을 높여줌.
"""
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from numpy.linalg import norm

embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-nli',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
)

print(embeddings_model)

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