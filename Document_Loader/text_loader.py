from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
import os

# Load env vars
load_dotenv()
os.environ['HF_HOME'] = './huggingface_cache'

# Define LLM
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 500,
    })

model = ChatHuggingFace(llm=llm)

loader = TextLoader("Document_Loader/RAG.txt", encoding="utf-8")
doc = loader.load()

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write the one line summary about following {text}",
    input_variables=["text"],
)

chain = prompt | model | parser

result = chain.invoke({'text' : doc[0].page_content})

print(result)