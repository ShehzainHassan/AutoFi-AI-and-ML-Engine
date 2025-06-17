from fastapi.testclient import TestClient
from app.main import app
import pytest
client = TestClient(app)
def test_get_recommendations():
	response = client.get('/api/recommendations/user/1')
	assert response.status_code == 200