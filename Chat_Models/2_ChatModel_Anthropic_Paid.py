from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(
    model_name="claude-3-sonnet-20240220",
    timeout=120,
    stop=None
)

result = model.invoke("What is the capital of France?", temperature=0.7, max_completion_tokens=100)

print("Chat Model Result:", result)