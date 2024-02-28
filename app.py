from fastapi import FastAPI
from langserve import add_routes
from llm_chain_with_rag import build_chain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="med_assist",
    version="0.1.0",
    description="Medical assistant API"
)

chain = build_chain()

add_routes(
    app=app,
    runnable=chain,
    path="/med_assist"
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)