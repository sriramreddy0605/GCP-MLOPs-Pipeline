from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from google.cloud import storage

app = Flask(__name__)

def load_model():
    try:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "hpc-toolkit-dev")
        bucket_name = f"{project}-ml-metadata"
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("models/model.joblib")
        blob.download_to_filename("/tmp/model.joblib")
        return joblib.load("/tmp/model.joblib")
    except:
        return None

model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None: model = load_model()
    if model is None: return jsonify({"error": "Model not ready in GCS"}), 500
    data = request.get_json()
    prediction = model.predict(np.array(data["input"]).reshape(1, -1))
    return jsonify({"prediction": int(prediction[0])})

@app.route("/", methods=["GET"])
def health():
    return "API is Live"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)