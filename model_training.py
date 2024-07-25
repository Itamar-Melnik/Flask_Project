import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
import pickle
from car_data_prep import prepare_data

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Process the data using the prepare_data function
processed_df = prepare_data(dataset)

# Separate features from the target variable
X = processed_df.drop(columns=['Price'])
y = processed_df['Price']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

# Fit the preprocessor and transform the data
X_processed = preprocessor.fit_transform(X)

# Create and train the model
model = ElasticNet(alpha=0.001, l1_ratio=0.6, random_state=42,max_iter=10000)
model.fit(X_processed, y)

# Create a dictionary to store both the preprocessor and the model
model_components = {'preprocessor': preprocessor,'model': model,}

# Creata a pickle file
pickle.dump(model_components, open("trained_model.pkl", "wb"))
