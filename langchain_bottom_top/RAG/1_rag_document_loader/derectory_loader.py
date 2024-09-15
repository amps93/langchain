# pipenv install unstructured
# unstructured: PDF, XML, HTML 등 미리 정의된 형식이 없는 텍스트 문서를 전처리하고 구조화된 형식으로 변환
# DirectoryLoader를 사용할 때 문서를 읽고 처리하기 위해 UnstructuredLoader가 내부적으로 사용됨

import os
from glob import glob

files = glob(os.path.join('./', '*.txt'))
print(files)

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(path='./', glob='*.txt')

data = loader.load()

print(len(data))
print(data[0])

