from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from google.cloud import storage

app = Flask(__name__)

# Expert Move: Pull model from GCS instead of local folder
def load_model():
    try:
        PROJECT_ID = os.environ.get('PROJECT_ID', 'hpc-toolkit-dev')
        BUCKET_NAME = f"{PROJECT_ID}-ml-metadata"
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob('models/model.joblib')
        blob.download_to_filename('/tmp/model.joblib')
        return joblib.load('/tmp/model.joblib')
    except Exception as e:
        print(f"Model download failed: {e}")
        return None

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not available in bucket'}), 500
    data = request.get_json()
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

@app.route('/', methods=['GET'])
def health():
    return "MLOps API is Live and Pulling from GCS!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
