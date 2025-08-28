import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

def load_model_and_encoders():
    """Load the trained model and encoders"""
    model = joblib.load('models/rf_model.pkl')
    encoders = joblib.load('models/encoders.pkl')
    return model, encoders

def preprocess_new_data(df, encoders):
    """Preprocess new data using saved encoders"""
    # Make a copy of the dataframe
    processed_df = df.copy()
    
    # Encode categorical features
    categorical_features = ['protocol_type', 'service', 'flag']
    for feature in categorical_features:
        if feature in processed_df.columns and feature in encoders:
            le = encoders[feature]
            # Handle unknown categories
            processed_df[feature] = processed_df[feature].apply(
                lambda x: x if x in le.classes_ else le.classes_[0])
            processed_df[feature] = le.transform(processed_df[feature])
    
    # Scale numerical features if scaler exists
    if 'scaler' in encoders:
        numerical_cols = processed_df.select_dtypes(include=[np.number]).columns
        processed_df[numerical_cols] = encoders['scaler'].transform(processed_df[numerical_cols])
    
    return processed_df

def predict(df):
    """Make predictions on new data"""
    model, encoders = load_model_and_encoders()
    
    # Preprocess data
    X = preprocess_new_data(df, encoders)
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Convert numeric predictions back to labels if needed
    if 'class' in encoders:
        y_pred_labels = encoders['class'].inverse_transform(y_pred)
    else:
        y_pred_labels = ['anomaly' if p == 1 else 'normal' for p in y_pred]
    
    return y_pred, y_pred_proba, y_pred_labels

def log_feedback(prediction_id, actual_label, predicted_label, correct):
    """Log user feedback on predictions"""
    os.makedirs('logs', exist_ok=True)
    log_file = 'logs/feedback_log.csv'
    
    # Create log file with headers if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('timestamp,prediction_id,actual_label,predicted_label,correct\n')
    
    # Append feedback
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{timestamp},{prediction_id},{actual_label},{predicted_label},{correct}\n')
    
    print(f"Feedback logged for prediction {prediction_id}")

def analyze_feedback():
    """Analyze feedback to evaluate model performance"""
    log_file = 'logs/feedback_log.csv'
    
    if not os.path.exists(log_file):
        return {
            'total_feedback': 0,
            'accuracy': 0,
            'feedback_by_class': {},
            'confusion_matrix': [[0, 0], [0, 0]]
        }
    
    feedback_df = pd.read_csv(log_file)
    
    # Calculate metrics
    total = len(feedback_df)
    correct = feedback_df['correct'].sum()
    accuracy = correct / total if total > 0 else 0
    
    # Feedback by class
    feedback_by_class = feedback_df['actual_label'].value_counts().to_dict()
    
    # Simple confusion matrix
    normal_as_normal = len(feedback_df[(feedback_df['actual_label'] == 'normal') & 
                                      (feedback_df['predicted_label'] == 'normal')])
    normal_as_anomaly = len(feedback_df[(feedback_df['actual_label'] == 'normal') & 
                                       (feedback_df['predicted_label'] == 'anomaly')])
    anomaly_as_normal = len(feedback_df[(feedback_df['actual_label'] == 'anomaly') & 
                                       (feedback_df['predicted_label'] == 'normal')])
    anomaly_as_anomaly = len(feedback_df[(feedback_df['actual_label'] == 'anomaly') & 
                                        (feedback_df['predicted_label'] == 'anomaly')])
    
    confusion_matrix = [[normal_as_normal, normal_as_anomaly], 
                        [anomaly_as_normal, anomaly_as_anomaly]]
    
    return {
        'total_feedback': total,
        'accuracy': accuracy,
        'feedback_by_class': feedback_by_class,
        'confusion_matrix': confusion_matrix
    }