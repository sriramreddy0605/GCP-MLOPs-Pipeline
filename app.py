from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model we trained in Phase 1
try:
    model = joblib.load('model.joblib')
except:
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not found. Please train the model first.'}), 500
    
    data = request.get_json()
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return jsonify({'prediction': int(prediction[0])})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
