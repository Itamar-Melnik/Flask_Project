import pandas as pd
import numpy as np
from datetime import datetime
import re

def prepare_data(dataset):  
    dataset = dataset[['manufactor','model','Year','Km','Engine_type','capacity_Engine','Gear','Price']] 

    # Replacing 'None' with NaN
    dataset = dataset.replace('None', np.nan)  

    # Standardizing categorical values
    dataset['manufactor'] = dataset['manufactor'].replace('Lexsus', 'לקסוס')  
    dataset['model'] = dataset['model'].replace(r'r/n', '', regex=True)   
    dataset['model'] = dataset['model'].str.replace(r'\(\d+\)', '', regex=True)  
    dataset['model'] = dataset['model'].replace(r'[\r\n()]', '', regex=True)
    dataset['model'] = dataset['model'].str.strip()  # Trimming spaces from model names
    dataset['model'] = dataset.apply(lambda row: re.sub(re.escape(row['manufactor']), '', row['model']).strip(), axis=1)

    dataset['Gear'] = dataset['Gear'].replace('אוטומט', 'אוטומטית')
    dataset['Gear'] = dataset['Gear'].replace('לא מוגדר', np.nan)

    dataset['Engine_type'] = dataset['Engine_type'].replace('היבריד', 'היברידי')


    # Cleaning numeric columns
    dataset['Km'] = dataset['Km'].replace(',', '')
    dataset['Km'] = pd.to_numeric(dataset['Km'], errors='coerce').astype('Int64')

    dataset['capacity_Engine'] = dataset['capacity_Engine'].replace(',', '')
    dataset['capacity_Engine'] = pd.to_numeric(dataset['capacity_Engine'], errors='coerce').astype('Int64')

    
    # Function to convert engine capacity
    def convert_capacity(x):
        if pd.isna(x):
            return x    
        if 100 <= x < 250:
            return x * 10
        elif x >= 10000:
            return x // 10
        else:
            return x

    dataset['capacity_Engine'] = dataset['capacity_Engine'].apply(convert_capacity)

    # Replace outliers with median
    median_value = dataset.capacity_Engine.median()
    dataset['capacity_Engine'] = dataset['capacity_Engine'].apply(lambda x: median_value if pd.notna(x) and (x < 800 or x > 6000) else x)
    dataset['capacity_Engine'] = pd.to_numeric(dataset['capacity_Engine'], errors='coerce').astype('Int64')

    dataset['Km'] = dataset.groupby('Year')['Km'].transform(lambda x: x.where((x >= 5000) & (x <= 500000), x.median()))

    # Fill missing values with mode
    fill_na_columns = ['capacity_Engine', 'Engine_type', 'Gear', 'Km']
    group_columns = [
        ['manufactor', 'model', 'Year'],
        ['manufactor', 'model'],
        ['model', 'Year'],
        ['manufactor', 'Year'],
        ['model'],
        ['manufactor']
    ]

    for col in fill_na_columns:
        for group in group_columns:
            dataset[col] = dataset.groupby(group)[col].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x))

    # Handle missing values that remained 
    for column in dataset.columns:
        if dataset[column].dtype == 'object':
            mode_value = dataset[column].mode().iloc[0]
            dataset[column].fillna(mode_value, inplace=True)
        else:
            median_value = dataset[column].median()
            dataset[column].fillna(median_value, inplace=True)
            
    # Create 'Age_Squared' feature
    dataset['Age_Squared'] = (datetime.now().year - dataset['Year'])**2  # Extracting year from 'Date' column

    # Rank categorical features
    gear_ranking = {
        'אוטומטית': 1,
        'טיפטרוניק': 2,
        'ידנית': 3,
        'רובוטית': 4
    }

    engine_type_ranking = {
        'חשמלי': 1,
        'היברידי': 2,
        'בנזין': 3,
        'טורבו דיזל': 4,
        'דיזל': 4,
        'גז': 5,
    }

    dataset['Gear'] = dataset['Gear'].map(gear_ranking)
    dataset['Engine_type'] = dataset['Engine_type'].map(engine_type_ranking)

    prepared_df = dataset.copy()

    return prepared_df       
