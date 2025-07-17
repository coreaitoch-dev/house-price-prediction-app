# app.py

import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# Create flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the form data as a list of floats
    float_features = [float(x) for x in request.form.values()]

    # Convert features to a NumPy array for the model
    features = [np.array(float_features)]

    # Make a prediction and format it
    prediction = model.predict(features)[0]
    # The dataset prices are in units of $100,000
    predicted_price = prediction * 100000

    return render_template("index.html", prediction_text=f"The predicted house price is ${predicted_price:,.2f}")


if __name__ == "__main__":
    app.run(debug=True)