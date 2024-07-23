import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
import pickle
from car_data_prep import prepare_data

# Load the dataset
dataset = pd.read_csv('dataset.csv')
processed_df = prepare_data(dataset)

# Separate features from the target variable
X = processed_df.drop(columns=['Price'])
y = processed_df['Price']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Create a ColumnTransformer to handle different types of features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'), categorical_features)
    ])

# Fit the transformer on the data and transform the data
X_processed = preprocessor.fit_transform(X)

# Train the model
model = ElasticNet(alpha=0.001, l1_ratio=0.8, random_state=42)
model.fit(X_processed, y)

# Save the model and transformers as pickle files
pickle.dump(model, open("trained_model.pkl", "wb"))
pickle.dump(preprocessor, open("preprocessor.pkl", "wb"))
