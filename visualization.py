import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib

def get_traffic_distribution_chart(df):
    """Generate traffic distribution chart as a pie chart"""
    plt.figure(figsize=(10, 6))
    
    if 'class' in df.columns:
        class_counts = df['class'].value_counts()
        
        # Create pie chart
        plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                startangle=90, shadow=False, explode=[0.05] * len(class_counts),
                textprops={'fontsize': 12})
        
        plt.title('Network Traffic Distribution', fontsize=16)
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        
        # Add legend for better readability
        plt.legend(class_counts.index, loc="best")
    else:
        plt.text(0.5, 0.5, 'No class data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return plot_data

def get_protocol_distribution_chart(df):
    """Generate protocol distribution chart with percentages"""
    plt.figure(figsize=(10, 6))
    
    if 'protocol_type' in df.columns:
        protocol_counts = df['protocol_type'].value_counts()
        total = protocol_counts.sum()
        
        # Create bar chart with better colors
        ax = sns.barplot(x=protocol_counts.index, y=protocol_counts.values, palette="viridis")
        
        # Add count and percentage labels
        for i, p in enumerate(ax.patches):
            percentage = f'{100 * p.get_height() / total:.1f}%'
            count = int(p.get_height())
            ax.annotate(f'{count}\n({percentage})', 
                        (p.get_x() + p.get_width() / 2., p.get_height()/2),
                        ha='center', va='center', fontsize=11,
                        color='white', weight='bold')
        
        plt.title('Traffic by Protocol', fontsize=16)
        plt.xlabel('Protocol', fontsize=12)
        plt.ylabel('Count', fontsize=12)
    else:
        plt.text(0.5, 0.5, 'No protocol data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return plot_data

def get_service_distribution_chart(df):
    """Generate service distribution chart as horizontal bar chart"""
    plt.figure(figsize=(12, 8))
    
    if 'service' in df.columns:
        service_counts = df['service'].value_counts().head(10)  # Top 10 services
        
        # Create horizontal bar chart
        ax = sns.barplot(y=service_counts.index, x=service_counts.values, palette="Blues_d")
        
        # Add count labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{int(p.get_width())}', 
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center', xytext=(5, 0), 
                        textcoords='offset points')
        
        plt.title('Top 10 Network Services', fontsize=16)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Service', fontsize=12)
    else:
        plt.text(0.5, 0.5, 'No service data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return plot_data

def get_confusion_matrix_chart():
    """Generate confusion matrix visualization with annotations"""
    plt.figure(figsize=(8, 6))
    
    try:
        cm = np.load('models/confusion_matrix.npy')
        
        # Calculate percentages for annotations
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = cm / row_sums
        annotations = np.empty_like(cm, dtype=str)
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f"{cm[i, j]}\n({norm_cm[i, j]:.1%})"
        
        # Create heatmap with improved styling
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'], 
                    yticklabels=['Normal', 'Anomaly'])
        
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
    except:
        plt.text(0.5, 0.5, 'Confusion matrix not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return plot_data

def get_feature_importance_chart():
    """Generate feature importance chart as horizontal bar chart"""
    plt.figure(figsize=(12, 8))
    
    try:
        feature_importance = pd.read_csv('models/feature_importance.csv')
        top_features = feature_importance.head(15)  # Top 15 features
        
        # Create horizontal bar chart with color gradient based on importance
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        ax = sns.barplot(x='importance', y='feature', data=top_features, palette=list(reversed(colors)))
        
        # Add importance value labels
        for i, p in enumerate(ax.patches):
            ax.annotate(f'{p.get_width():.3f}', 
                        (p.get_width(), p.get_y() + p.get_height() / 2),
                        ha='left', va='center', xytext=(5, 0), 
                        textcoords='offset points')
        
        plt.title('Top 15 Feature Importance', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
    except:
        plt.text(0.5, 0.5, 'Feature importance data not available', 
                 horizontalalignment='center', verticalalignment='center')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return plot_data

def get_detection_timeline(df, time_column=None):
    """Generate enhanced timeline of detections with trend line"""
    plt.figure(figsize=(14, 6))
    
    # If no time column is provided, create one
    if time_column is None or time_column not in df.columns:
        # Use count column to simulate time
        if 'count' in df.columns:
            df = df.sort_values('count')
            time_values = range(len(df))
        else:
            time_values = range(len(df))
    else:
        time_values = df[time_column]
    
    if 'class' in df.columns:
        # Create binary values (1 for anomaly, 0 for normal)
        binary_class = df['class'].apply(lambda x: 1 if x != 'normal' else 0)
        
        # Create a scatter plot with better visual distinction
        plt.scatter(time_values, binary_class, c=binary_class, cmap='coolwarm', 
                   alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
        
        # Add a rolling average line to show trend
        window_size = min(50, len(binary_class))
        if window_size > 1:
            rolling_avg = binary_class.rolling(window=window_size, center=True).mean()
            plt.plot(time_values, rolling_avg, 'g-', linewidth=2, alpha=0.7, 
                    label=f'Rolling average (window={window_size})')
        
        plt.yticks([0, 1], ['Normal', 'Anomaly'])
        plt.title('Network Traffic Anomaly Detection Timeline', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Traffic Type', fontsize=12)
        plt.legend()
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No class data available', 
                 horizontalalignment='center', verticalalignment='center')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return plot_data