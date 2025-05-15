from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_similar_responses_endpoint():
    # Test with a valid question
    response = client.post(
        "/similar_responses",
        json={"question": "What is the capital of France?"}
    )
    
    # Check status code and response structure
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)

def test_similar_responses_invalid_request():
    # Test with missing required field
    response = client.post(
        "/similar_responses",
        json={}
    )
    assert response.status_code == 422  # Unprocessable Entity
