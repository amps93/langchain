"""
PDF 문서 페이지별로 로드
pipenv isntall -q pypdf
"""
from langchain_community.document_loaders import PyPDFLoader

pdf_filepath = '000660_SK_2023.pdf'
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

print('----------PDF 문서 페이지별로 로드----------')
print(len(pages))
print(pages[10].page_content[:30])

"""
형식이 없는 PDF 문서 로드 (UnstructuredPDFLoader)
pipenv install unstructured unstructured-inference
오류1. pdf2image.exceptions.PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?
해결 - https://velog.io/@tett_77/pdf2image.exceptions.PDFInfoNotInstalledError-Unable-to-get-page-count.-Is-poppler-installed-and-in-PATH
오류2. unstructured_pytesseract.pytesseract.TesseractNotFoundError: tesseract is not installed or it's not in your PATH. See README file for more information.
해결 - https://najakprogram.tistory.com/8
"""
from langchain_community.document_loaders import UnstructuredPDFLoader
import unstructured_pytesseract

pdf_filepath = '000660_SK_2023.pdf'
unstructured_pytesseract.pytesseract.tesseract_cmd = r"C:\Users\amps\AppData\Local\tesseract.exe"

# 전체 텍스트를 단일 문서 객체로 변환
loader = UnstructuredPDFLoader(pdf_filepath)
pages = loader.load()

print('----------형식이 없는 PDF 문서 로드----------')
print(len(pages))
print(pages[0].page_content[:50])
print(pages[0].metadata)  # PDF 문서의 원본 출처 등 메타 데이터 확인

# 각 테스트 조각을 별도로 분리 ( mode="elements")
# UnstructuredPDFLoader는 내부적으로 각 텍스트를 별도의 요소(element)로 생성합니다.
# load 메소드를 호출할 때 mode="elements" 옵션을 설정하면 텍스트 청크들이 서로 분리된 상태로 유지할 수 있습니다.
# 즉 각 요소들이 별도의 Document 객체로 변환됩니다.
pdf_filepath = '000660_SK_2023.pdf'

# 텍스트 조각(chunk)를 별도 문서 객체로 변환
loader = UnstructuredPDFLoader(pdf_filepath, mode='elements')
pages = loader.load()

print(len(pages))
print(pages[100])
print(pages[101])
print(pages[102])

"""
PDF 문서의 메타 데이터를 상세하게 추출 (PyMuPDFLoader)
pipenv install pymupdf
"""
print('----------PDF 문서의 메타 데이터를 상세하게 추출----------')
from langchain_community.document_loaders import PyMuPDFLoader

pdf_filepath = '000660_SK_2023.pdf'

loader = PyMuPDFLoader(pdf_filepath)
pages = loader.load()

print(len(pages))
print(pages[0].page_content)
print(pages[0].metadata)


"""
온라인(on-line) PDF 문서 로드 (OnlinePDFLoader)
"""
print('----------온라인(on-line) PDF 문서 로드----------')
from langchain_community.document_loaders import OnlinePDFLoader

# Transformers 논문을 로드
loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
pages = loader.load()

print(len(pages))
print(pages[0].page_content[:200])

"""
특정 폴더의 모든 PDF 문서 로드 (PyPDFDirectoryLoader)
"""
print('----------특정 폴더의 모든 PDF 문서 로드----------')
from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader('./')
data = loader.load()

print(len(data))
print(data[0])
print(data[-3])
