from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
text = """
The error you’re encountering is a KeyError from LangChain, specifically indicating that the PromptTemplate in your chain is expecting certain input variables (notes and quiz), but the input provided to the chain only includes text. This mismatch is causing the chain to fail during execution. Let’s break down the issue and explain how to fix it.
"""


loader = PyPDFLoader('./Document_Loader/data/Multimodol Complete Literature.pdf')

doc = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator='')

result = splitter.split_documents(doc)

print(result[0].page_content)