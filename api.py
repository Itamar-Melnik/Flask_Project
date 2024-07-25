import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
from datetime import datetime
import pandas as pd
from car_data_prep import prepare_data

app = Flask(__name__)

# Load the pickle file containing both model and preprocessor
model_components = pickle.load(open('trained_model.pkl', 'rb'))

model = model_components['model']
preprocessor = model_components['preprocessor']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get input features from the form
    features = {
        'manufactor': request.form.get('manufactor'),
        'model': request.form.get('model'),
        'Year': int(request.form.get('Year')),
        'Km': int(request.form.get('Km')),
        'Engine_type': request.form.get('Engine_type'),
        'capacity_Engine': float(request.form.get('capacity_Engine')),
        'Gear': request.form.get('Gear'),
        'Price': 0  # Placeholder for Price
    }

    # Create a DataFrame from the input features
    input_df = pd.DataFrame([features])

    # Use prepare_data function to preprocess the input
    processed_df = prepare_data(input_df)
    
    # Remove the 'Price' column from the processed data
    X = processed_df.drop(columns=['Price'])

    # Apply the preprocessor to the input data
    X_processed = preprocessor.transform(X)

    # Make prediction
    prediction = model.predict(X_processed)[0]

    # Format the output
    output_text = f"המחיר המשוער לרכב זה הוא {int(prediction)} ש״ח"

    return render_template('index.html', prediction_text=output_text, **request.form)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

