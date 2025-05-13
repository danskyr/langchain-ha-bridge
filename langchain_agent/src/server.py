from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI

# from langchain.chains.router import RouterChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

app = FastAPI()

# Get configuration from environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
router_model = os.environ.get("ROUTER_MODEL", "gpt-3.5-turbo")
query_model = os.environ.get("QUERY_MODEL", "gpt-4")

# Init LLMs
router_llm = ChatOpenAI(
    temperature=0.0,
    model_name=router_model,
    openai_api_key=openai_api_key
)

query_llm = ChatOpenAI(
    temperature=0.7,
    model_name=query_model,
    openai_api_key=openai_api_key
)

# Define the device-control chain
device_chain = LLMChain(
    llm=router_llm,
    prompt=PromptTemplate(
        input_variables=["command"],
        template="""
You are an IoT controller. Interpret the user command and output a Home Assistant service call payload in YAML.
User Command: {command}
Output only the YAML service call.
""",
    ),
)

# Define a general query chain
general_chain = LLMChain(
    llm=query_llm,
    prompt=PromptTemplate(
        input_variables=["question"],
        template="""
You are a helpful assistant. Answer the following question in clear, concise language.
Question: {question}
""",
    ),
)

# Setup a simple router (RouterChain API has changed)
# For now, we'll just use a simple function to route requests
def route_request(text):
    # In a real implementation, this would use the router_llm to determine the route
    if "turn on" in text.lower() or "turn off" in text.lower():
        return device_chain.run(command=text)
    else:
        return general_chain.run(question=text)

class Req(BaseModel):
    text: str

class Resp(BaseModel):
    response: str

@app.post("/process", response_model=Resp)
def process(req: Req):
    """Accepts {"text": "..."}; returns {"response": "..."}."""
    out = route_request(req.text)
    return {"response": out}

# This allows the server to be run directly with `python server.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
