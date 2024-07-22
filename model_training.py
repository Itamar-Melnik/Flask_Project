import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from car_data_prep import prepare_data
import pickle

dataset = pd.read_csv('dataset.csv')
processed_df = prepare_data(dataset)

X = processed_df.drop(columns=['Price'])
y = processed_df['Price']

#Encoding the data
categorical_features = X.select_dtypes(include=['object'])
numerical_features = X.select_dtypes(exclude=['object'])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
X_categorical = encoder.fit_transform(categorical_features)

#Scaling the data
scaler = StandardScaler()
X_numerical = scaler.fit_transform(numerical_features)

X_processed = np.hstack((X_numerical, X_categorical))

#Training the model
model = ElasticNet(alpha=0.01, l1_ratio=0.9, random_state=42)
model.fit(X_processed, y)

#Creating pickels to api
pickle.dump(model, open("trained_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
