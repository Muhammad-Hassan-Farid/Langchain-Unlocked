import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, load_prompt

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

st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

# Load Template
template = load_prompt('Prompts/prompts_template.json')

# Fill Place Holder
prompt = template.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
        })


if st.button('Summarize'):
    result  = model.invoke(prompt)
    st.write(result.content)

