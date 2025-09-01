from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
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

# Define schema
class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

# Parser
parser = PydanticOutputParser(pydantic_object=Person)

# Template (force JSON values, not schema)
template = PromptTemplate(
    template=(
        "Generate the name, age, and city of a fictional {place} person.\n"
        "Respond ONLY in valid JSON with this format:\n"
        "{format_instruction}\n\n"
        "Example:\n"
        '{"name": "Arjun Silva", "age": 29, "city": "Colombo"}'
    ),
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# Chain
chain = template | model | parser

# Run
try:
    final_result = chain.invoke({'place': 'Sri Lankan'})
    print("✅ Parsed:", final_result)
except Exception as e:
    print("⚠️ Failed to parse. Raw error:", e)
