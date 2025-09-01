# This code will return the embedding of a query e.g "What is the capital of France?" -> Embedding -> [2.3, 0.1, -1.2, ...]

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = ["What is the capital of France?",
            "What is the largest mammal in the world?",
            "What is the boiling point of water?"]

query = "What is the capital of France?"

doc_embedding = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)[0] 

print(list(enumerate(scores)))

index, score = sorted(enumerate(scores), key=lambda x: x[1])[-1]

print(f"Most similar document: {documents[index]} (Score: {score})")
