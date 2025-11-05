import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib

def load_and_preprocess_data():
    """Load and preprocess the crime dataset"""
    df = pd.read_csv('data/raw/crime_tn_dataset.csv')
    
    # Create additional features
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    df['is_festival_season'] = df['festival_period']
    df['crime_rate_index'] = (df['past_crime_rate'] * df['population_density']) / 1000
    
    # Encode categorical variables
    le_district = LabelEncoder()
    le_area = LabelEncoder()
    le_crime = LabelEncoder()
    le_day = LabelEncoder()
    le_time = LabelEncoder()
    le_weather = LabelEncoder()
    le_age = LabelEncoder()
    
    df['district_encoded'] = le_district.fit_transform(df['district'])
    df['area_type_encoded'] = le_area.fit_transform(df['area_type'])
    df['crime_type_encoded'] = le_crime.fit_transform(df['crime_type'])
    df['day_of_week_encoded'] = le_day.fit_transform(df['day_of_week'])
    df['time_of_day_encoded'] = le_time.fit_transform(df['time_of_day'])
    df['weather_encoded'] = le_weather.fit_transform(df['weather'])
    df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
    
    # Save encoders
    encoders = {
        'district': le_district,
        'area_type': le_area,
        'crime_type': le_crime,
        'day_of_week': le_day,
        'time_of_day': le_time,
        'weather': le_weather,
        'age_group': le_age
    }
    joblib.dump(encoders, 'models/encoders.pkl')
    
    # Select features for modeling
    feature_cols = [
        'district_encoded', 'area_type_encoded', 'latitude', 'longitude',
        'month', 'day_of_week_encoded', 'time_of_day_encoded', 'population_density',
        'unemployment_rate', 'literacy_rate', 'poverty_index', 'police_station_count',
        'cctv_density', 'past_crime_rate', 'weather_encoded', 'festival_period',
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
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
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
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    load_and_preprocess_data()