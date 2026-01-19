# Used Car Price Prediction - Flask Web Application

This repository deploys a machine learning model for predicting used car prices as a simple web application.

## What's Inside
- `api.py`: Flask API server for loading the model and handling predictions.
- `car_data_prep.py`: Function for data preprocessing.
- `model_training.py`: Script for training and saving the ElasticNet model.
- `index.html`: Basic HTML interface for user input.
- `trained_model.pkl`: Pre-trained model file.
- `requirements.txt`: Python dependencies.
- `Team members.txt`: Project team details.

## Project Goal
Deploy a trained ML model (from car data analysis) as a user-friendly web app where users input car details to get price predictions.

## Technologies
- Flask (web framework)
- scikit-learn (modeling)
- pandas, numpy (data handling)
- HTML/CSS (frontend)

## How to Run
1. Clone the repo: `git clone https://github.com/Itamar-Melnik/Flask_Project.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python api.py`
4. Open http://127.0.0.1:5000/ in your browser.

## Status
Academic project from July 2024. No active development; archived for reference.

## Related Projects
This is part of a larger car price prediction pipeline:
- [Project-web_scraping](https://github.com/Itamar-Melnik/Project-web_scraping): Data scraping and initial ML.
- [ML-project](https://github.com/Itamar-Melnik/ML-project): ML on messy dataset.
