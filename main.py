from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from agent import app as agent_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uuid
import os

api = FastAPI()

limiter = Limiter(key_func=get_remote_address)
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class QueryRequest(BaseModel):
    query: str
    thread_id: str = None


@api.post("/research")
@limiter.limit("15/minute")
def research(request: Request, body: QueryRequest, x_api_key: str = Header(...)):
    if x_api_key != os.getenv("AGENT_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        if body.thread_id:
            thread_id = body.thread_id
        else:
            thread_id = str(uuid.uuid4())
        initial_state = {
            "query": body.query,
            "search": [],
            "answer": "",
            "verdict": "",
            "reason": "",
            "retry_count": 0,
            "total_cost": 0.0
        }

        config = {"configurable": {"thread_id": thread_id}}
        result = agent_app.invoke(initial_state, config=config)
        return {"answer": result["answer"], "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")


@api.get("/health")
@limiter.limit("5/minute")
def health(request: Request):
    return {"status": "ok"}