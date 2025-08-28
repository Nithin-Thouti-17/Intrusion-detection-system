import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset(file_path='data/network_traffic.csv'):
    """Load the network traffic dataset"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Returning empty dataframe.")
            # Create a minimal dataframe with expected columns for visualization
            return pd.DataFrame({
                'protocol_type': [],
                'service': [],
                'flag': [],
                'class': []
            })
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Warning: {file_path} is empty. Returning empty dataframe.")
            return pd.DataFrame({
                'protocol_type': [],
                'service': [],
                'flag': [],
                'class': []
            })
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        # Return a minimal empty dataframe
        return pd.DataFrame({
            'protocol_type': [],
            'service': [],
            'flag': [],
            'class': []
        })

def preprocess_data(df):
    """Preprocess the network traffic data for model training"""
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Convert categorical features
    categorical_features = ['protocol_type', 'service', 'flag']
    encoders = {}
    
    for feature in categorical_features:
        if feature in processed_df.columns:
            le = LabelEncoder()
            processed_df[feature] = le.fit_transform(processed_df[feature])
            encoders[feature] = le
    
    # Handle the class label
    if 'class' in processed_df.columns:
        # Create binary classification: normal vs anomaly
        processed_df['binary_class'] = processed_df['class'].apply(
            lambda x: 'normal' if x == 'normal' else 'anomaly')
        
        # Encode the class labels
        le_class = LabelEncoder()
        processed_df['class_encoded'] = le_class.fit_transform(processed_df['class'])
        processed_df['binary_class_encoded'] = processed_df['binary_class'].apply(
            lambda x: 0 if x == 'normal' else 1)
        encoders['class'] = le_class
    
    # Scale numerical features
    numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols 
                      if col not in ['class_encoded', 'binary_class_encoded']]
    
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])
        encoders['scaler'] = scaler
    
    return processed_df, encoders

def analyze_traffic(df):
    """Analyze network traffic data for dashboard stats"""
    stats = {}
    
    # Count of normal vs anomaly traffic
    if 'class' in df.columns:
        stats['traffic_by_class'] = df['class'].value_counts().to_dict()
    
    # Traffic by protocol
    if 'protocol_type' in df.columns:
        stats['traffic_by_protocol'] = df['protocol_type'].value_counts().to_dict()
    
    # Traffic by service
    if 'service' in df.columns:
        stats['traffic_by_service'] = df['service'].value_counts().head(10).to_dict()
    
    # Traffic by flag
    if 'flag' in df.columns:
        stats['traffic_by_flag'] = df['flag'].value_counts().to_dict()
    
    return stats