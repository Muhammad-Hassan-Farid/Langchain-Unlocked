from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
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
    template='Generate a tweet post about {topic}',
    input_variables=['topic']
)

Prompt1 = PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel(
    {
        'tweet': RunnableSequence(Prompt, model, parser),
        'linkedin' : RunnableSequence(Prompt1, model, parser)
    }
)

result = parallel_chain.invoke({'topic': 'AI'})

print(result['tweet'])
print(result['linkedin'])