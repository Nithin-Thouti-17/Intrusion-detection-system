import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import base64
from datetime import datetime
import uuid
from io import StringIO
import matplotlib.pyplot as plt

# Import our modulesstreamlit run app.py
from data_processor import load_dataset, preprocess_data, analyze_traffic
from model_evaluator import predict, log_feedback, analyze_feedback
from visualization import (get_traffic_distribution_chart, get_protocol_distribution_chart,
                          get_service_distribution_chart, get_confusion_matrix_chart,
                          get_feature_importance_chart, get_detection_timeline)

# Page configuration
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🛡️ Network IDS")
page = st.sidebar.radio("Navigation", 
                      ["Dashboard", "Upload & Analyze", "Admin Feedback", "Analytics"])

# Check if model exists, if not, show warning
if not os.path.exists('models/rf_model.pkl'):
    st.sidebar.warning("⚠️ Model not found! Please run model_trainer.py first.")

# Load dataset for visualization
@st.cache_data
def get_dataset():
    try:
        df = load_dataset()
        if df.empty:
            st.warning("⚠️ Dataset is empty or not found! Please check the data/network_traffic.csv file.")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

# Main Dashboard Page
if page == "Dashboard":
    st.title("Network Intrusion Detection Dashboard")
    
    # Get data
    dataset = get_dataset()
    
    if dataset.empty:
        st.warning("No data available to display. Please ensure data/network_traffic.csv exists and contains data.")
        st.info("You can upload data in the 'Upload & Analyze' section.")
    else:
        # Continue with your existing dashboard code...
    
        # First row of stats
        col1, col2, col3 = st.columns(3)
        
        # Traffic stats
        traffic_stats = analyze_traffic(dataset)
        
        with col1:
            st.metric(
                label="Total Traffic Monitored", 
                value=f"{len(dataset):,}"
            )
        
        with col2:
            if 'traffic_by_class' in traffic_stats and 'normal' in traffic_stats['traffic_by_class']:
                normal_count = traffic_stats['traffic_by_class']['normal']
                st.metric(
                    label="Normal Traffic", 
                    value=f"{normal_count:,}",
                    delta=f"{normal_count/len(dataset)*100:.1f}%"
                )
        
        with col3:
            if 'traffic_by_class' in traffic_stats:
                anomaly_count = sum(v for k, v in traffic_stats['traffic_by_class'].items() if k != 'normal')
                st.metric(
                    label="Anomaly Traffic", 
                    value=f"{anomaly_count:,}",
                    delta=f"{anomaly_count/len(dataset)*100:.1f}% of total",
                    delta_color="inverse"
                )
        
        # Second row - charts
        st.subheader("Traffic Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            traffic_chart = get_traffic_distribution_chart(dataset)
            st.image(f"data:image/png;base64,{traffic_chart}")
        
        with col2:
            protocol_chart = get_protocol_distribution_chart(dataset)
            st.image(f"data:image/png;base64,{protocol_chart}")
        
        # Third row - more charts
        col1, col2 = st.columns(2)
        
        with col1:
            service_chart = get_service_distribution_chart(dataset)
            st.image(f"data:image/png;base64,{service_chart}")
        
        with col2:
            timeline_chart = get_detection_timeline(dataset)
            st.image(f"data:image/png;base64,{timeline_chart}")
        
        # Fourth row - details
        st.subheader("Recent Activity")
        
        # Show the most recent 10 records
        st.dataframe(dataset.tail(10))

# Upload & Analyze Page
elif page == "Upload & Analyze":
    st.title("Upload & Analyze Network Traffic")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload network traffic data (CSV)", type="csv")
    
    # Sample data option
    use_sample = st.checkbox("Or use sample data for testing")
    
    df = None
    
    if uploaded_file is not None:
        # Read the uploaded file
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Show a preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    elif use_sample and os.path.exists('data/test_sample.csv'):
        df = pd.read_csv('data/test_sample.csv')
        st.success(f"Using sample data with {df.shape[0]} rows")
        
        # Show a preview
        st.subheader("Sample Data Preview")
        st.dataframe(df.head())
    
    # Analysis section
    if df is not None:
        st.subheader("Run Analysis")
        
        if st.button("Analyze Traffic"):
            try:
                # Check if model exists
                if not os.path.exists('models/rf_model.pkl'):
                    st.error("Model not found! Please run model_trainer.py first.")
                else:
                    # Make predictions
                    with st.spinner("Analyzing traffic data..."):
                        # Drop the class column if it exists (we're predicting it)
                        X = df.drop(['class', 'binary_class', 'class_encoded', 'binary_class_encoded'],
                                    axis=1, errors='ignore')
                        
                        y_pred, y_pred_proba, y_pred_labels = predict(X)
                        
                        # Add predictions to dataframe
                        results = X.copy()
                        results['predicted_class'] = y_pred_labels
                        results['anomaly_probability'] = [prob[1] if len(prob) > 1 else prob[0] for prob in y_pred_proba]
                        
                        # Display results
                        st.subheader("Analysis Results")
                        
                        # Summary metrics
                        anomaly_count = sum(1 for label in y_pred_labels if label != 'normal')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Records Analyzed", len(results))
                        with col2:
                            st.metric("Anomalies Detected", 
                                      anomaly_count,
                                      delta=f"{anomaly_count/len(results)*100:.1f}%",
                                      delta_color="inverse")
                        
                        # Show detailed results
                        st.subheader("Detailed Results")
                        
                        # Sort by anomaly probability
                        results_sorted = results.sort_values('anomaly_probability', ascending=False)
                        
                        # Add styling to highlight anomalies
                        def highlight_anomalies(row):
                            return ['background-color: #ffcccc' if row['predicted_class'] != 'normal' else '' for _ in row]
                        
                        st.dataframe(results_sorted.style.apply(highlight_anomalies, axis=1))
                        
                        # Download results button
                        csv = results_sorted.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="analysis_results.csv">Download Results as CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        # Visualize the anomalies
                        st.subheader("Anomaly Detection Visualization")
                        
                        # Create a timeline of anomalies
                        fig_data = get_detection_timeline(
                            pd.DataFrame({'class': y_pred_labels})
                        )
                        st.image(f"data:image/png;base64,{fig_data}")
                        
                        # Save the analysis session
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        session_name = f"session_{timestamp}"
                        os.makedirs('analysis_sessions', exist_ok=True)
                        results_sorted.to_csv(f'analysis_sessions/{session_name}.csv', index=False)
                        st.success(f"Analysis session saved: {timestamp.replace('_', ' ')} with {len(results_sorted)} records and {anomaly_count} anomalies detected")
                        
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")


# Admin Feedback Page
elif page == "Admin Feedback":
    st.title("Admin Feedback")
    
    # Show recent analysis sessions
    st.subheader("Recent Analysis Sessions")
    
    sessions = []
    if os.path.exists('analysis_sessions'):
        for file in os.listdir('analysis_sessions'):
            if file.startswith('session_') and file.endswith('.csv'):
                session_id = file.replace('session_', '').replace('.csv', '')
                timestamp = datetime.fromtimestamp(os.path.getmtime(f'analysis_sessions/{file}'))
                
                # Load session data to get more info
                session_data = pd.read_csv(f'analysis_sessions/{file}')
                num_records = len(session_data)
                num_anomalies = sum(1 for _, row in session_data.iterrows() if row['predicted_class'] != 'normal')
                
                sessions.append({
                    'session_id': session_id,
                    'timestamp': timestamp,
                    'file_path': f'analysis_sessions/{file}',
                    'num_records': num_records,
                    'num_anomalies': num_anomalies,
                    'display_name': f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {num_records} records ({num_anomalies} anomalies)"
                })
    
    if sessions:
        sessions_df = pd.DataFrame(sessions)
        sessions_df = sessions_df.sort_values('timestamp', ascending=False)
        
        # Select a session with friendly display names
        selected_session_display = st.selectbox(
            "Select analysis session to review:",
            options=sessions_df['display_name'].tolist()
        )
        
        if selected_session_display:
            # Get the actual session ID from the display name
            selected_session = sessions_df[sessions_df['display_name'] == selected_session_display]['session_id'].iloc[0]
            
            # Get file path
            file_path = sessions_df[sessions_df['session_id']==selected_session]['file_path'].iloc[0]
            
            # Load session data
            session_data = pd.read_csv(file_path)
            
            # Display session data
            st.dataframe(session_data)
            
            # Allow feedback on specific rows
            st.subheader("Provide Feedback")
            row_number = st.number_input("Select row number to provide feedback", 
                                        min_value=0, max_value=len(session_data)-1, value=0)
            
            # Show selected row
            selected_row = session_data.iloc[row_number]
            st.write("Selected record:")
            st.json(selected_row.to_dict())
            
            # Feedback form
            actual_label = st.radio("Actual label:", ["normal", "anomaly"])
            prediction_correct = actual_label == selected_row['predicted_class']
            
            if st.button("Submit Feedback"):
                log_feedback(
                    prediction_id=f"{selected_session}_{row_number}",
                    actual_label=actual_label,
                    predicted_label=selected_row['predicted_class'],
                    correct=prediction_correct
                )
                
                st.success("Feedback submitted successfully!")
    else:
        st.info("No analysis sessions found. Run an analysis first.")
    
    # Show feedback summary
    st.subheader("Feedback Summary")
    feedback_stats = analyze_feedback()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Feedback Entries", feedback_stats['total_feedback'])
    
    with col2:
        if feedback_stats['total_feedback'] > 0:
            st.metric("Model Accuracy (Based on Feedback)", 
                     f"{feedback_stats['accuracy']:.2%}")
            
    # Display confusion matrix
    if feedback_stats['total_feedback'] > 0:
        st.subheader("Confusion Matrix (From Feedback)")
        cm = feedback_stats['confusion_matrix']
        cm_df = pd.DataFrame(cm, 
                           index=['Actual: Normal', 'Actual: Anomaly'],
                           columns=['Predicted: Normal', 'Predicted: Anomaly'])
        st.table(cm_df)

# Analytics Page
elif page == "Analytics":
    st.title("Network Traffic Analytics")
    
    # Get data
    dataset = get_dataset()
    
    # Show model performance
    st.header("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm_chart = get_confusion_matrix_chart()
        st.image(f"data:image/png;base64,{cm_chart}")
    
    with col2:
        # Feature importance
        st.subheader("Feature Importance")
        fi_chart = get_feature_importance_chart()
        st.image(f"data:image/png;base64,{fi_chart}")
    
    # Show traffic patterns
    st.header("Traffic Patterns")
    
    # Service distribution
    st.subheader("Service Distribution")
    service_chart = get_service_distribution_chart(dataset)
    st.image(f"data:image/png;base64,{service_chart}")
    
    # Add raw data exploration
    st.header("Data Exploration")
    
    if st.checkbox("Show raw data"):
        # Number of rows to display
        rows = st.slider("Number of rows", 5, 100, 10)
        st.dataframe(dataset.head(rows))
    
    # Column statistics
    if st.checkbox("Show column statistics"):
        st.write("Numeric columns statistics:")
        st.write(dataset.describe())
        
        st.write("Categorical columns statistics:")
        cat_cols = dataset.select_dtypes(include=['object']).columns
        for col in cat_cols:
            st.write(f"**{col}** value counts:")
            st.write(dataset[col].value_counts())
    
    # Correlation matrix section - moved inside the Analytics page block with proper indentation
    if st.checkbox("Show correlation matrix"):
        st.subheader("Feature Correlation Matrix")
        
        # Select only numeric columns
        numeric_df = dataset.select_dtypes(include=[np.number])
        
        # If there are too many columns, select only the top ones
        if numeric_df.shape[1] > 15:
            # Try to get feature importance if available
            try:
                feature_importance = pd.read_csv('models/feature_importance.csv')
                top_features = feature_importance['feature'].head(15).tolist()
                # Filter only the features that exist in the dataset
                available_features = [f for f in top_features if f in numeric_df.columns]
                numeric_df = numeric_df[available_features]
            except:
                # If feature importance not available, select first 15 columns
                numeric_df = numeric_df.iloc[:, :15]
        
        # Calculate and display correlation matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        import io
        
        # Create correlation matrix
        corr = numeric_df.corr()
        
        # Generate correlation heatmap with improved styling
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Better colormap and styling
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Create the heatmap with improved annotations
        heatmap = sns.heatmap(
            corr, 
            mask=mask, 
            cmap=cmap, 
            vmax=.9,  # Adjusted for better color range
            vmin=-.9,
            center=0,
            square=True, 
            linewidths=.8, 
            annot=True, 
            fmt='.2f',
            cbar_kws={"shrink": .8, "label": "Correlation Coefficient"}
        )
        
        # Improve the appearance
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        
        # Add explanatory note if desired
        plt.figtext(0.5, 0.01, 
                   "Values closer to +1 or -1 indicate stronger correlations", 
                   ha="center", fontsize=10, 
                   bbox={"facecolor":"lightgrey", "alpha":0.3, "pad":5})
        
        # Convert to image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_data = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        st.image(f"data:image/png;base64,{img_data}")
        
        # Add some interpretation text
        st.info("""
            **How to interpret:** 
            * Values close to +1 indicate strong positive correlation (when one feature increases, the other also increases)
            * Values close to -1 indicate strong negative correlation (when one feature increases, the other decreases)
            * Values near 0 indicate little to no linear correlation
        """)

if __name__ == "__main__":
    # Display a footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **AI-Powered Intrusion Detection System**
        
        This application uses machine learning to detect 
        potential network intrusions. For more information,
        see the project documentation.
        """
    )