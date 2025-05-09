from fastapi import FastAPI
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains.router import RouterChain
from langchain import LLMChain
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

# Setup RouterChain
router = RouterChain.from_llm(
    llm=router_llm,
    chains={"device": device_chain, "query": general_chain},
    default_chain=general_chain,
)

class Req(BaseModel):
    text: str

class Resp(BaseModel):
    response: str

@app.post("/process", response_model=Resp)
def process(req: Req):
    """Accepts {"text": "..."}; returns {"response": "..."}."""
    out = router.run(req.text)
    return {"response": out}

# This allows the server to be run directly with `python server.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
