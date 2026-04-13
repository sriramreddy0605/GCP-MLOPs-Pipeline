from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from google.cloud import storage

app = Flask(__name__)

def load_model():
    try:
        # Get project ID from environment or default
        project = os.environ.get('GOOGLE_CLOUD_PROJECT', 'hpc-toolkit-dev')
        bucket_name = f"{project}-ml-metadata"
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob('models/model.joblib')
        
        # Download to /tmp (Cloud Run allows writing here)
        blob.download_to_filename('/tmp/model.joblib')
        return joblib.load('/tmp/model.joblib')
    except Exception as e:
        print(f"Model download failed: {e}")
        return None

# Attempt to load model at startup
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        model = load_model()  # Try loading again if it failed at startup
    
    if model is None:
        return jsonify({'error': 'Model still not available in GCS'}), 500
        
    data = request.get_json()
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

@app.route('/', methods=['GET'])
def health():
    return "MLOps API: Connected to GCS"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
