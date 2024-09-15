from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='한국주택금융공사_주택금융관련_지수_20160101.csv', encoding='cp949')
data = loader.load()

print(len(data))
print(data[0])

"""
데이터 출처 정보를 특정 필드(열, column)로 지정 - source_column 
"""
loader = CSVLoader(file_path='한국주택금융공사_주택금융관련_지수_20160101.csv', encoding='cp949',
                   source_column='연도')
data = loader.load()
print(data[0])

"""
CSV 파싱 옵션을 지정 - csv_args 
"""
loader = CSVLoader(file_path='한국주택금융공사_주택금융관련_지수_20160101.csv', encoding='cp949',
                   csv_args={
                       'delimiter': '\n',
                   })

data = loader.load()
print(data[0])