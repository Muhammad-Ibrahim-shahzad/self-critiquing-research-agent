from dotenv import load_dotenv
from typing import TypedDict
from groq import Groq
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class ResearchState(TypedDict):
    query: str
    search:  list[dict]
    answer: str
    verdict: str
    reason: str
    retry_count: int