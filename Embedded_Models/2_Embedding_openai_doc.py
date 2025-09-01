# This code will return the embedding of a query e.g "What is the capital of France?" -> Embedding -> [2.3, 0.1, -1.2, ...]

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

document = ["What is the capital of France?",
            "What is the largest mammal in the world?",
            "What is the boiling point of water?"]

result = embedding.embed_documents(document)

print(str(result))  
 