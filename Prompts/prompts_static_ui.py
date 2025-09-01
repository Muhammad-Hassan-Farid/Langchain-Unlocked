import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

import os

os.environ['HF_HOME'] = '/Users/hassanfarid/Documents/Langchain GenAI/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation', 
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 100,
    })

model = ChatHuggingFace(llm=llm)

st.header('Research Tool')


user_input = st.text_input("Ask a question:")

if st.button('Summarize'):
    response = model.invoke(user_input)
    st.write(response.content)

