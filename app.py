from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model trained in experiment.py
try:
    model = joblib.load('model.joblib')
except Exception as e:
    print(f"Model load failed: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    # Expecting JSON: {"input": [1, 2, 3, 4]}
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

@app.route('/', methods=['GET'])
def health():
    return "MLOps API is Live!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
