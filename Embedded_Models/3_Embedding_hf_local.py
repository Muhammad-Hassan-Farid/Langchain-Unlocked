import os
os.environ["USE_TF"] = "0"

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document = ["What is the capital of France?",
            "What is the largest mammal in the world?",
            "What is the boiling point of water?"]


result = embedding.embed_documents(document)

print(str(result))