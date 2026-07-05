from fastapi.testclient import TestClient
from src.api.main import app

def test_predict():
    print("Starting API Test Client...")
    client = TestClient(app)
    
    payload = {
        "text": "नेपाल राष्ट्र बैंकले नयाँ नीति जारी गरेको छ।",
        "source_type": "official",
        "category": "finance"
    }
    
    print("Sending POST request to /predict...")
    response = client.post("/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Success! API returned: {data}")
    else:
        print(f"Failed! Status: {response.status_code}, Response: {response.text}")

if __name__ == "__main__":
    test_predict()
