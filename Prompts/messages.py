from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import os

os.environ['HF_HOME'] = '/Users/hassanfarid/Documents/Langchain GenAI/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation', 
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 500,
    })

model = ChatHuggingFace(llm=llm)


messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me about Langchain.")
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)