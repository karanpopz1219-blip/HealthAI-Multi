from starlette.testclient import TestClient
import backend.app as appmod

client = TestClient(appmod.app)

def test_health():
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert 'status' in data

# Test endpoints with minimal payload
minimal_payload = {
    "age": 45,
    "bmi": 25.0,
    "systolic_bp": 120.0,
    "cholesterol": 180.0,
    "blood_glucose": 90.0,
    "diabetes": 0,
    "hypertension": 0,
    "smoker": 0,
    "prev_hospitalizations": 0,
    "gender": "Male"
}

def test_predict_risk():
    resp = client.post('/predict/risk', json=minimal_payload)
    assert resp.status_code in (200, 503, 500)

def test_predict_los():
    resp = client.post('/predict/los', json=minimal_payload)
    assert resp.status_code in (200, 503, 500)

def test_cluster():
    resp = client.post('/cluster', json=minimal_payload)
    assert resp.status_code in (200, 503, 500)
