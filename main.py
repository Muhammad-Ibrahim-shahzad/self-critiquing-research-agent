from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import app as agent_app

api = FastAPI()

class QueryRequest(BaseModel):
    query: str


@api.post("/research")
def research(request: QueryRequest):
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
    except HTTPException as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")