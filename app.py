from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)
model = load('model.joblib')  # Make sure the model is in the same directory

@app.route('/')
def home():
    return "Welcome to the Iris Classifier API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Expecting data in JSON format
    features = data['features']
    prediction = model.predict([features])
    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
