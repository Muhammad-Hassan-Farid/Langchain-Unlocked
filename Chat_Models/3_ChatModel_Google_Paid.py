from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

result = model.invoke("What is the capital of France?", temperature=0.7, max_completion_tokens=100)

print("Chat Model Result:", result)

print("Chat Model Name:", model.name)

print("Chat Model Result:", result.content)