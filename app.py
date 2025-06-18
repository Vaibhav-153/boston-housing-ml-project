# Import necessary libraries
import pickle  # For loading saved model and scaler
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np

# Create a Flask web app instance
app = Flask(__name__)

# Load the saved linear regression model and scaler
regmodel = pickle.load(open("lr_model.pkl", "rb"))  # Trained Linear Regression model
ss = pickle.load(open("scaling.pkl", "rb"))         # Standard Scaler used for input features

# Route for homepage (HTML form input)
@app.route('/')
def home():
    return render_template('home.html')  # Loads the home.html form page

# Route to handle API predictions (JSON input)
@app.route('/predict_api', methods=['POST'])  # Use 'POST' not 'Post'
def predict_api():
    # Get data from JSON request
    data = request.json['data']
    print("Received Data (JSON):", data)

    # Convert data to array and reshape for model input
    input_array = np.array(list(data.values())).reshape(1, -1)
    print("Input as Numpy Array:", input_array)

    # Apply same scaling as training data
    new_data = ss.transform(input_array)

    # Make prediction
    output = regmodel.predict(new_data)
    print("Predicted Output (House Price):", output[0])

    # Return prediction as JSON
    return jsonify({"predicted_price": output[0]})

# Route to handle form submission (HTML input)
@app.route('/predict', methods=['POST'])
def predict():
    # Extract values from form and convert to float
    data = [float(x) for x in request.form.values()]
    print("Received Form Data:", data)

    # Scale the input data
    final_input = ss.transform(np.array(data).reshape(1, -1))
    print("Scaled Input for Model:", final_input)

    # Predict using the model
    output = regmodel.predict(final_input)[0]

    # Return the prediction result to the same HTML page
    return render_template("home.html", 
                           prediction_text=f"The predicted Median House Value (in $1000s) is: {round(output, 2)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)  # Starts Flask app in debug mode
