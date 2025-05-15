import pytest
import numpy as np
from src.retriever.retriever import string_to_array, generate_answer, get_similar_responses

def test_string_to_array():
    # Test normal case
    array_str = "[0.1 0.2 0.3 0.4]"
    result = string_to_array(array_str)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4
    assert np.allclose(result, np.array([0.1, 0.2, 0.3, 0.4]))
    
    # Test error handling
    bad_str = "not an array"
    result = string_to_array(bad_str)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 384  # Default dimension

def test_generate_answer():
    # Basic test with mock context
    question = "What is the population of Maklavan?"
    contexts = ["Maklavan is a city in Iran with a population of 2,170 individuals in 2006."]
    answer = generate_answer(question, contexts)
    
    # Should return a non-empty string
    assert isinstance(answer, str)
    assert len(answer) > 0
    
    # Should contain the population number
    assert "2,170" in answer or "2170" in answer

def test_get_similar_responses_structure():
    # Test the structure of the response
    question = "What is the capital of France?"
    result = get_similar_responses(question)
    
    # Check that the result has the expected structure
    assert isinstance(result, dict)
    assert "generated_answer" in result
    assert "supporting_contexts" in result
    assert isinstance(result["generated_answer"], str)
    assert isinstance(result["supporting_contexts"], list)




