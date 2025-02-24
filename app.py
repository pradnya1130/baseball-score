from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model (1).pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from formp
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = f'Predicted Baseball Score: {prediction[0]}'

    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
