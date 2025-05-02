from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_identityFeedbackscore():
    response = client.post("/identityfeedbackscore", json={"key": "value"})
    assert response.status_code == 200
    assert "result" in response.json()
