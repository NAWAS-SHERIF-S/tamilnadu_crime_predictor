import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def load_and_preprocess_data():
    """Load and preprocess the crime dataset"""
    df = pd.read_csv('data/raw/crime_tn_dataset_new.csv')
    
    # Create additional features
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_festival_season'] = df['festival_period']
    df['crime_rate_index'] = (df['past_crime_rate'] * df['population_density']) / 1000
    
    # Encode categorical variables
    le_district = LabelEncoder()
    le_taluk = LabelEncoder()
    le_area = LabelEncoder()
    le_crime = LabelEncoder()
    le_day = LabelEncoder()
    le_time = LabelEncoder()
    # le_weather = LabelEncoder()  # Removed weather
    le_age = LabelEncoder()
    
    df['district_encoded'] = le_district.fit_transform(df['district'])
    df['taluk_encoded'] = le_taluk.fit_transform(df['taluk'])
    df['area_type_encoded'] = le_area.fit_transform(df['area_type'])
    df['crime_type_encoded'] = le_crime.fit_transform(df['crime_type'])
    df['day_of_week_encoded'] = le_day.fit_transform(df['day_of_week'])
    df['time_of_day_encoded'] = le_time.fit_transform(df['time_of_day'])
    # df['weather_encoded'] = le_weather.fit_transform(df['weather'])  # Removed weather
    df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
    
    # Save encoders
    encoders = {
        'district': le_district,
        'taluk': le_taluk,
        'area_type': le_area,
        'crime_type': le_crime,
        'day_of_week': le_day,
        'time_of_day': le_time,
        # 'weather': le_weather,  # Removed weather
        'age_group': le_age
    }
    joblib.dump(encoders, 'models/encoders.pkl')
    
    # Select features for modeling
    feature_cols = [
        'district_encoded', 'taluk_encoded', 'area_type_encoded', 'latitude', 'longitude',
        'month', 'day_of_week_encoded', 'time_of_day_encoded', 'population_density',
        'unemployment_rate', 'literacy_rate', 'poverty_index', 'police_station_count',
        'cctv_density', 'past_crime_rate', 'festival_period',
        'age_group_encoded', 'gender_ratio', 'road_density', 'education_index',
        'internet_penetration', 'alcohol_availability', 'transport_access',
        'public_event', 'is_weekend', 'crime_rate_index'
    ]
    
    X = df[feature_cols]
    y = df['crime_type_encoded']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Use 100% data for training (no test/validation split)
    X_train = X_scaled
    y_train = y
    X_val = X_scaled  # Same as training for 100% usage
    X_test = X_scaled  # Same as training for 100% usage
    y_val = y
    y_test = y
    
    # Save processed data
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/y_test.npy', y_test)
    
    # Optional PCA
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    joblib.dump(pca, 'models/pca.pkl')
    np.save('data/processed/X_train_pca.npy', X_train_pca)
    np.save('data/processed/X_val_pca.npy', X_val_pca)
    np.save('data/processed/X_test_pca.npy', X_test_pca)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"PCA features: {X_train_pca.shape[1]}")
    print(f"Training set: {X_train.shape[0]} samples (100% of data)")
    print(f"Validation set: {X_val.shape[0]} samples (same as training)")
    print(f"Test set: {X_test.shape[0]} samples (same as training)")
    
    # Train a quick model and generate evaluation plots
    generate_evaluation_plots(X_train, X_test, y_train, y_test, encoders['crime_type'])
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def generate_evaluation_plots(X_train, X_test, y_train, y_test, crime_encoder):
    """Generate confusion matrix and ROC curves"""
    
    # Train a logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    crime_labels = crime_encoder.classes_
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=crime_labels, yticklabels=crime_labels)
    plt.title('Confusion Matrix - Crime Type Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Crime Type', fontsize=12)
    plt.ylabel('Actual Crime Type', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. ROC Curves (One-vs-Rest for multiclass)
    plt.figure(figsize=(12, 10))
    
    # Calculate ROC curve for each class
    n_classes = len(crime_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i, (crime_type, color) in enumerate(zip(crime_labels, colors)):
        # Create binary labels for current class
        y_test_binary = (y_test == i).astype(int)
        y_score = y_pred_proba[:, i]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{crime_type} (AUC = {roc_auc:.2f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Crime Type Prediction (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Classification Report
    report = classification_report(y_test, y_pred, target_names=crime_labels)
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(report)
    
    # Save classification report
    with open('reports/classification_report.txt', 'w') as f:
        f.write("Classification Report - Crime Type Prediction\n")
        f.write("="*50 + "\n")
        f.write(report)
    
    print("\nEvaluation plots saved to 'reports/' directory:")
    print("   - confusion_matrix.png")
    print("   - roc_curves.png")
    print("   - classification_report.txt")

if __name__ == "__main__":
    print("Starting data preprocessing and model evaluation...")
    load_and_preprocess_data()
    print("\nPreprocessing completed with evaluation plots generated!")