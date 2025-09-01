from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert."),
    ('human', "Explain {topic} in simple words")
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': 'Outswing'})

print(prompt)