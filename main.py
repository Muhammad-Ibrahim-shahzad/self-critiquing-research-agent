from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from agent import app as agent_app
import os

api = FastAPI()

class QueryRequest(BaseModel):
    query: str


@api.post("/research")
def research(request: QueryRequest, x_api_key: str = Header(...)):
    if x_api_key != os.getenv("AGENT_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        initial_state = {
            "query": request.query,
            "search": [],
            "answer": "",
            "verdict": "",
            "reason": "",
            "retry_count": 0
        }
        result = agent_app.invoke(initial_state)
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")