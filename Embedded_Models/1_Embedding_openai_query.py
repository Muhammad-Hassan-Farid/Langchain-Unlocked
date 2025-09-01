# This code will return the embedding of a query e.g "What is the capital of France?" -> Embedding -> [2.3, 0.1, -1.2, ...]

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

result = embedding.embed_query("What is the capital of France?")  

print(str(result))  
