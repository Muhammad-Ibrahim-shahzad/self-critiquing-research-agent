from dotenv import load_dotenv
from typing import TypedDict
from groq import Groq
import json
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

system_prompt = """
You are a critique agent. Evaluate the draft answer below.
Respond with ONLY raw JSON, no extra text, no markdown formatting, in this
exact structure:
{
   "verdict": "GOOD",
   "reason": "The provided data is complete",
   "ref_query": null
}
verdict must be EXACTLY one of these two strings: "GOOD" or "NEEDS_IMPROVEMENT"
Additionally, check whether the draft answer contains any claims that are NOT 
supported by the search results — facts, numbers, names, or dates, not present 
in state["search"] — mark the verdict as NEEDS_IMPROVEMENT and explain in the 
reason field exactly which claim is unsupported by the sources.
Here are example responses:
Example 1:
{
   "verdict": "NEEDS_IMPROVEMENT",
   "reason": "The provided data is incomplete",
   "ref_query": "Todays bitcoin price"
}
Example 2:
{
   "verdict": "GOOD",
   "reason": "No improvement needed",
   "ref_query": null
} 
"""

def critique_node(state: ResearchState) -> dict:
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Draft Answer: {state['answer']}\n\nSearch results: {state['search']}"}
    ]

    response = client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        messages=messages
    )

    raw_text = response.choices[0].message.content

    try:
        critique_dict = json.loads(raw_text)
    except json.JSONDecodeError:
        critique_dict = {
            "verdict": "NEEDS_IMPROVEMENT",
            "reason": "Critique response was not valid JSON",
            "ref_query": state["query"]
        }

    return {
        "verdict": critique_dict["verdict"],
        "reason": critique_dict["reason"],
        "query": critique_dict["ref_query"] if critique_dict["ref_query"] else state["query"]
    }

def should_retry(state: ResearchState) -> str:

    if state["verdict"] == "NEEDS_IMPROVEMENT" and state["retry_count"] < 2:
        return  "retry"
    else:
        return "done"