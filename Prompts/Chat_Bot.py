from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, load_prompt
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


chat_history = [
    SystemMessage(content="You are a helpful AIassistant.")
]  
    
while True:
    user_input = input("You: ")

    chat_history.append(HumanMessage(content=user_input))
    
    if user_input.lower() == 'exit':
        break
    result = model.invoke(chat_history)

    chat_history.append(AIMessage(content=result.content))
    print("Bot:", result.content)
    

print("Chat History:", chat_history)
