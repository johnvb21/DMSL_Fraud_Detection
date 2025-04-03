import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

def process_fraud_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    # Drop missing rows
    df = df.dropna()

    # Convert types
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['is_fraud'] = df['is_fraud'].astype(int)

    # Feature: Age at transaction time
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365

    # Time features
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    df['transaction_date'] = df['trans_date_trans_time'].dt.date

    # Geospatial feature: Distance between user and merchant (Haversine)
    df['distance'] = haversine_vectorized(
        df['lat'], df['long'],
        df['merch_lat'], df['merch_long']
    )

    # Drop irrelevant or ID-based columns
    drop_cols = ['trans_date_trans_time', 'dob', 'trans_num', 'first', 'last', 'street', 'transaction_date']
    df = df.drop(columns=drop_cols)

    # Encode categorical features
    categorical_cols = ['merchant', 'category', 'gender', 'state', 'city', 'job']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


def oversample(X, y):
    smote = SMOTE(random_state=42)
    X_new, y_new = smote.fit_resample(X, y)
    return X_new, y_new
