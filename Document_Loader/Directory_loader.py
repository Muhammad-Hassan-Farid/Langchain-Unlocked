from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

load = DirectoryLoader(
    path="Document_Loader/data",
    glob="*.pdf",
    loader_cls = PyPDFLoader
)

# docs = load.load()

docs = load.lazy_load()

print(docs)