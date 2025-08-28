import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_processor import load_dataset, preprocess_data

def train_model():
    """Train the intrusion detection model and save it"""
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load dataset
    df = load_dataset()
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Preprocess data
    processed_df, encoders = preprocess_data(df)
    print("Data preprocessing completed")
    
    # Save encoders for later use
    joblib.dump(encoders, 'models/encoders.pkl')
    
    # Prepare features and target
    X = processed_df.drop(['class', 'binary_class', 'class_encoded', 'binary_class_encoded'], 
                          axis=1, errors='ignore')
    y = processed_df['binary_class_encoded'] if 'binary_class_encoded' in processed_df.columns else processed_df['class_encoded']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save confusion matrix data
    cm = confusion_matrix(y_test, y_pred)
    np.save('models/confusion_matrix.npy', cm)
    
    # Extract feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('models/feature_importance.csv', index=False)
    print("Top 10 important features:")
    print(feature_importance.head(10))
    
    # Save the model
    joblib.dump(rf_model, 'models/rf_model.pkl')
    print("Model saved successfully!")
    
    # Save a sample of test data for demonstration
    test_sample = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1).sample(100)
    test_sample.to_csv('data/test_sample.csv', index=False)
    
    return rf_model, encoders

if __name__ == "__main__":
    train_model()