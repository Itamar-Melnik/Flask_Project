# Used Car Price Prediction — Flask Web Application

This repository contains the model deployment part of a used car price prediction project — a simple Flask web app that serves predictions using a pre-trained machine learning model.

## What's inside

- **`api.py`**  
  Main Flask application — loads the trained model and exposes a prediction endpoint + basic web interface.

- **`car_data_prep.py`**  
  Data preprocessing function (same logic used during training).

- **`model_training.py`**  
  Script that trains the model (ElasticNet regression) and saves it as `trained_model.pkl`.

- **`index.html`**  
  Simple HTML frontend for users to input car details and get a predicted price.

- **`trained_model.pkl`**  
  Pre-trained model file (ready to use).

- **`requirements.txt`**  
  List of Python dependencies needed to run the app.

- **`Team members.txt`**  
  Project team information.

## How to run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
