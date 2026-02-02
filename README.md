# Used Car Price Prediction – Part 3: Flask Web Application

**Academic end-to-end ML project (2024) – Part 3/3**

Production-ready web application that serves live price predictions for used cars using the trained ElasticNet model.

---

## 📋 Project Overview

End-to-end deployment of the used car price prediction model:
- **User input** → car details (year, km, manufacturer, model, engine type, etc.)
- **Flask backend** → preprocesses input using shared `prepare_data()` pipeline
- **ML model** → ElasticNet predicts price
- **Real-time result** → displayed instantly to user

This repository demonstrates MLOps practices including model serialization, API design, and web deployment.

### Data Flow Context
- **Part 1:** Scraped 54 Peugeot records from ad.co.il
- **Part 2:** Trained model on instructor's combined dataset (all manufacturers)
- **Part 3:** Deployed model serves predictions for any manufacturer/model

## ✨ Features

- **Real-time predictions** via REST API
- **Input validation** for all features
- **Shared preprocessing pipeline** (ensures training-serving consistency)
- **Simple web interface** for end-user interaction
- **Serialized model** for efficient loading

## 📁 Repository Structure

```
.
├── api.py                 # Flask server & prediction endpoint
├── car_data_prep.py       # Shared preprocessing pipeline (same as training)
├── model_training.py      # Script to train & serialize ElasticNet model
├── templates/
│   └── index.html        # Simple HTML input form
├── models/
│   └── trained_model.pkl # Serialized ElasticNet model
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠️ Technologies

- **Flask** – Lightweight web framework
- **scikit-learn** – ElasticNet model & preprocessing
- **pandas / numpy** – Data manipulation
- **pickle** – Model serialization
- **HTML + CSS** – Frontend interface

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Itamar-Melnik/used-car-price-prediction-03-deployment.git
cd used-car-price-prediction-03-deployment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model (optional – pre-trained model included)
```bash
python model_training.py
```

### 4. Run the Flask server
```bash
python api.py
```

### 5. Access the application
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

## 🌐 API Endpoints

### `GET /`
Returns the HTML form for user input.

### `POST /predict`
Accepts car details and returns predicted price.

**Request Body (JSON):**
```json
{
  "year": 2019,
  "km": 45000,
  "manufacturer": "Toyota",
  "model": "Corolla",
  "hand": 2,
  "engine_type": "Petrol",
  "engine_capacity": 1600,
  "area": "Center",
  "city": "Tel Aviv",
  "color": "White",
  "supply_score": 150,
  "test_days": 200,
  "photo_count": 8
}
```

**Response (JSON):**
```json
{
  "predicted_price": 85000,
  "currency": "ILS"
}
```

## 🔧 How It Works

1. **User submits form** with car details
2. **Flask receives data** via POST request
3. **Preprocessing pipeline** (`car_data_prep.py` - `prepare_data()` function) transforms input:
   - Same transformations as training phase (ensures consistency)
   - Categorical encoding
   - Feature scaling
   - Missing value handling
4. **Model predicts** using `trained_model.pkl`
5. **Result returned** to user interface

### Key Design Decision
The `prepare_data()` function in `car_data_prep.py` is **shared** between:
- `model_training.py` (Part 2) - for training data preparation
- `api.py` (Part 3) - for live prediction preprocessing

This ensures **training-serving consistency** and prevents prediction drift.

## 📊 Model Details

- **Algorithm:** ElasticNet Regression
- **Training data:** Combined dataset from all teams (all manufacturers) with intentional errors introduced by instructor
- **Original scraping:** 54 Peugeot records (Part 1)
- **Features:** 20+ including year, km, manufacturer, engine type, supply score
- **Validation:** 10-fold cross-validation
- **Performance:** RMSE optimized during training
- **Preprocessing:** Shared `prepare_data()` function ensures consistency between training and serving

## 🔗 Full Pipeline Overview

| Stage | Repository | Description |
|-------|------------|-------------|
| 1 | [used-car-price-prediction-01-scraping](https://github.com/Itamar-Melnik/used-car-price-prediction-01-scraping) | Web scraping real listings |
| 2 | [used-car-price-prediction-02-ml](https://github.com/Itamar-Melnik/used-car-price-prediction-02-ml) | Cleaning messy data & model training |
| **3** | **[used-car-price-prediction-03-deployment](https://github.com/Itamar-Melnik/used-car-price-prediction-03-deployment)** | **Flask web app for live prediction (this repo)** |

## 🎓 Key Learnings

- **Model serialization** with pickle
- **Training-serving consistency** via shared preprocessing
- **REST API design** for ML models
- **Flask web development** basics
- **End-to-end ML deployment** workflow

## 📝 License

MIT License

**Academic project (July 2024) · No active maintenance**

---

*This repository is the most viewed part of the pipeline – showcases model serving & basic MLOps practices.*
