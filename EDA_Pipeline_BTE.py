import pandas as pd
import numpy as np

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


def bayesian_target_encode(df, categorical_cols, target_col, alpha):
    """
    Perform Bayesian Target Encoding on categorical features with smoothing.

    Parameters:
    df : DataFrame
        The dataset
    categorical_cols : list
        List of categorical columns to encode.
    target_col : int
        The target variable (is_fraud)
    alpha : float
        Smoothing factor

    Returns:
    df_encoded : DataFrame
        A copy of the original DataFrame with target-encoded columns.
    """
    global_mean = df[target_col].mean()
    df_encoded = df.copy()

    for col in categorical_cols:
        # Compute category-specific mean and size
        category_mean = df.groupby(col)[target_col].mean()
        category_size = df.groupby(col).size()

        # Apply Bayesian smoothing
        smoothed_mean = (category_mean * category_size + global_mean * alpha) / (category_size + alpha)

        # Map encoded values back to the dataset
        df_encoded[col] = df_encoded[col].map(smoothed_mean)

    return df_encoded


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
    drop_cols = ['trans_date_trans_time', 'dob', 'trans_num', 'first', 'last', 'street', 'transaction_date','zip']
    df = df.drop(columns=drop_cols)

    # One-hot encoding for 'gender' and 'state'
    df = pd.get_dummies(df, columns=['gender', 'state'])
    
    # Regular Target Encoding for merchant and category because there are no underrepresented classes 
        # Calculate the means
    merch_mean = df.groupby('merchant')['is_fraud'].mean()
    cat_mean = df.groupby('category')['is_fraud'].mean()
        # Substitute means into columns
    df['merchant'] = df['merchant'].map(merch_mean)
    df['category'] = df['category'].map(cat_mean)
    
    # Apply Bayesian Target Encoding to high cardinality features with underrepresented classes
    BTE_cols = ['city','job',"cc_num"]
    df = bayesian_target_encode(df, BTE_cols, target_col='is_fraud', alpha=10)

    # Convert Boolean Values to Integers
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)
    
    return df
