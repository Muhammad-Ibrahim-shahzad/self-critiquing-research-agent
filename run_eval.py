from eval_dataset import golden_dataset
from agent import app

def run_eval():
    correct = 0
    total = len(golden_dataset)

    for entry in golden_dataset:
        initial_state = {
            "query": entry["query"],
            "search": [],
            "answer": "",
            "verdict": "",
            "reason": "",
            "retry_count": 0
        }
        result = app.invoke(initial_state)
        answer = result["answer"]

        any_of_check = any(keyword in answer for keyword in entry["expected_any_of"]) if entry.get("expected_any_of") else True
        is_correct = all(keyword in answer for keyword in entry["expected_keywords"]) and any_of_check

        if is_correct:
            correct += 1
        else:
            print(f"Failed: {entry['query']}")
            print(f"Got: {answer}\n")

    print(f"Score: {correct}/{total}")

if __name__ == "__main__":
    run_eval()