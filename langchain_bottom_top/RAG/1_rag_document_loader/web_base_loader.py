"""
web_paths: 로드할 웹 페이지의 URL을 단일 문자열 또는 여러 개의 URL을 시퀀스 배열로 지정할 수 있음
bs_kwargs: BeautifulSoup을 사용하여 HTML을 파싱할 때 사용되는 인자들을 딕셔너리 형태로 제공
"""
import bs4
from langchain_community.document_loaders import WebBaseLoader


# 여러 개의 url 지정 가능
url1 = "https://blog.langchain.dev/few-shot-prompting-to-improve-tool-calling-performance/"
url2 = "https://blog.langchain.dev/improving-core-tool-interfaces-and-docs-in-langchain/"

loader = WebBaseLoader(
    web_paths=(url1, url2),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("article-header", "article-content")
        )
    ),
)
docs = loader.load()
print(len(docs))
print(docs[0].page_content[:100])  #첫 번째 문서의 내용 100글자 출력
