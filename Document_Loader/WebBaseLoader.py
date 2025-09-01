from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load environment variables
load_dotenv()

# Define LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Load webpage
url = 'https://www.w3schools.com/java/java_data_types.asp'
loader = WebBaseLoader(url)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(docs)

# Parser
parser = StrOutputParser()

# Prompt
prompt = PromptTemplate(
    template="Answer the following question:\n{question}\nBased on the following text:\n{text}",
    input_variables=["question", "text"],
)

# Create chain
chain = prompt | model | parser

# Run chain with first chunk of text
result = chain.invoke({
    "question": "What is data types?",
    "text": docs
})

print(result)
