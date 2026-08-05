from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from dotenv import load_dotenv
from typing import TypedDict
from groq import Groq
import time
import json
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

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

def research_node(state: ResearchState) -> dict:

    start = time.time()
    query = state["query"]
    search_results = tavily_client.search(query=query)
    results = search_results["results"]

    end = time.time()
    print(f"Time taken by research node: {end - start}")

    return {"search": results}

def draft_node(state: ResearchState) -> dict:

    start = time.time()
    draft_system_prompt="""
    You are a research assistant, answer the question using the search results below.
    """
    messages = [
        {"role": "system",
         "content": draft_system_prompt},
        {"role": "user",
         "content": f"Search results: {state['search']}\n\nOriginal Query: {state['query']}"}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    answer = response.choices[0].message.content

    input_cost = (response.usage.prompt_tokens/1_000_000)*0.59
    output_cost = (response.usage.completion_tokens/1_000_000)*0.79
    total_cost = input_cost + output_cost

    end = time.time()
    print(f"Time taken by Draft node: {end - start}")
    print(f"Cost of Draft node: ${total_cost:.6f}")

    return {"answer": answer}

#separation of concerns(merge_query, parse_critique)

def merge_query(ref_query, current_query): #helper function
    return ref_query if ref_query else current_query

def parse_critique(raw_text, current_query): #helper function
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "verdict": "NEEDS_IMPROVEMENT",
            "reason": "Critique response was not valid JSON",
            "ref_query": current_query
        }

def critique_node(state: ResearchState) -> dict:

    start = time.time()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Draft Answer: {state['answer']}\n\nSearch results: {state['search']}"}
    ]

    response = client.chat.completions.create(
        model= "llama-3.3-70b-versatile",
        messages=messages
    )

    raw_text = response.choices[0].message.content

    critique_dict = parse_critique(raw_text, state["query"])

    end = time.time()
    print(f"Time taken by Critique node: {end - start}")

    return {
        "verdict": critique_dict["verdict"],
        "reason": critique_dict["reason"],
        "query": merge_query(critique_dict["ref_query"], state["query"]),
        "retry_count": state["retry_count"] + 1
    }

def should_retry(state: ResearchState) -> str:

    if state["verdict"] == "NEEDS_IMPROVEMENT" and state["retry_count"] < 2:
        return  "retry"
    else:
        return "done"
    
graph = StateGraph(ResearchState)

#nodes
graph.add_node("research_node", research_node)
graph.add_node("draft_node", draft_node)
graph.add_node("critique_node", critique_node)

#edges
graph.set_entry_point("research_node")
graph.add_edge("research_node", "draft_node")
graph.add_edge("draft_node", "critique_node")
graph.add_conditional_edges(
    "critique_node",
    should_retry,
    {
        "retry": "research_node",
        "done": END
    }
)

app = graph.compile()
if __name__ == "__main__":

    initial_state = {
        "query": input("What do you want to search? "),
        "search": [],
        "answer": "",
        "verdict": "",
        "reason": "",
        "retry_count": 0
    }
    
    start = time.time()
    accumulated_state = initial_state.copy()
    
    for step in app.stream(initial_state):
        print(step)
        for node_name, node_output in step.items():
            accumulated_state.update(node_output)
    
    print(f"\n\n{accumulated_state['answer']}")

    end = time.time()
    print(f"Total time taken: {end - start}")