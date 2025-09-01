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

Prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

Prompt1 = PromptTemplate(
    template='Explain the fowwing joke {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = RunnableSequence(Prompt, model, parser, Prompt1 , model ,parser)

print(chain.invoke({'topic': 'programming'}))

