import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler

# Function for extracting mappings
def Extract_Mappings(train_df, BTE_cols, TE_cols, target_col='is_fraud', alpha=10):
    """
    Automatically extract Bayesian Target Encoding (BTE) and Regular Target Encoding (TE) mappings.
    """
    mappings = {}
    global_mean = train_df[target_col].mean()

    # Regular Target Encoding
    TE_maps = {
        col: train_df.groupby(col)[target_col].mean().to_dict()
        for col in TE_cols
    }

    # Bayesian Target Encoding
    BTE_maps = {}
    for col in BTE_cols:
        category_mean = train_df.groupby(col)[target_col].mean()
        category_size = train_df.groupby(col).size()
        smoothed_mean = (category_mean * category_size + global_mean * alpha) / (category_size + alpha)
        BTE_maps[col] = smoothed_mean.to_dict()

    mappings['TE_maps'] = TE_maps
    mappings['BTE_maps'] = BTE_maps
    mappings['global_mean'] = global_mean
    return mappings


# Function for substituting values (TE and BTE in test data)
def Apply_Encodings(df, mappings, BTE_cols, TE_cols):
    """
    Apply precomputed encodings to test data.
    """
    global_mean = mappings['global_mean']

    # Regular Target Encoding
    for col in TE_cols:
        df[col] = df[col].map(mappings['TE_maps'][col]).fillna(global_mean)

    # Bayesian Target Encoding
    for col in BTE_cols:
        df[col] = df[col].map(mappings['BTE_maps'][col]).fillna(global_mean)

    return df


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


def process_test_data(file_path, mappings, BTE_cols, TE_cols):
    """
    Process and encode test data consistently with training mappings.
    """
    # Load the test data
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.dropna()
    df = df.drop(columns=['is_fraud'], errors='ignore')

    # Feature engineering
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
    df['transaction_date'] = df['trans_date_trans_time'].dt.date
    df['distance'] = haversine_vectorized(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )

    # Drop irrelevant columns
    drop_cols = ['trans_date_trans_time', 'dob', 'trans_num', 'first', 'last', 'street', 'transaction_date', 'zip']
    df = df.drop(columns=drop_cols)

    # One-hot encoding for categorical features
    df = pd.get_dummies(df, columns=['gender'])

    # Apply encodings
    df = Apply_Encodings(df, mappings, BTE_cols, TE_cols)

    # Convert Boolean Values to Integers
    bool_cols = df.select_dtypes(include=['bool']).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Normalize the features and convert back to DataFrame for consistency
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    # Convert to sparse matrix
    Test_Prepped_Data = csr_matrix(df.values)

    return Test_Prepped_Data

