from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

os.environ['HF_HOME'] = '/Users/hassanfarid/Documents/Langchain GenAI/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation', 
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 200,
    })

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "Generate 5 interesting one line facts about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke({"topic": "Python Programming"})

print(result)

chain.get_graph().print_ascii()