from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct")

Result = llm.invoke("What is the capital of France?")

print("LLM Result:", Result)
