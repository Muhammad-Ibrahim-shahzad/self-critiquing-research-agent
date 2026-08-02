from agent import should_retry, merge_query, parse_critique
import json

def test_should_retry_when_needs_improvement():
    fake_test = {"verdict": "NEEDS_IMPROVEMENT", "retry_count": 0}
    result = should_retry(fake_test)
    assert result == "retry"

def test_should_retry_when_good():
    fake_state = {"verdict": "GOOD", "retry_count": 0}
    result = should_retry(fake_state)
    assert result == "done"

def test_should_retry_when_limit_reached():
    fake_state = {"verdict": "NEEDS_IMPROVEMENT", "retry_count": 2}
    result = should_retry(fake_state)
    assert result == "done"

#merge query test

def test_merge_query_when_ref_query_exists():
    result = merge_query(ref_query="10 days forecast Lahore", current_query="Weather Lahore")
    assert result == "10 days forecast Lahore"

def test_merge_query_when_ref_query_is_none():
    result = merge_query(ref_query=None, current_query="Weather Lahore")
    assert result == "Weather Lahore"

def test_parse_critique_with_valid_json():
    raw_text = json.dumps({"verdict": "GOOD", "reason": "All good", "ref_query": None})
    result = parse_critique(raw_text, current_query="original query")
    assert result["verdict"] == "GOOD"

def test_parse_critique_with_invalid_json():
    raw_text = "this is not a valid json{{"
    result = parse_critique(raw_text, current_query="original query")
    assert result["verdict"] == "NEEDS_IMPROVEMENT"
    assert result["ref_query"] == "original query"