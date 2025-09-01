from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./Document_Loader/data/Multimodol Complete Literature.pdf')

doc = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator='')

result = splitter.split_documents(doc)

print(result[0].page_content)