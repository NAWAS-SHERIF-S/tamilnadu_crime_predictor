from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg

app = Flask(__name__)

# Load model and preprocessors
model = joblib.load('models/crime_model.pkl')
encoders = joblib.load('models/encoders.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load original dataset for dashboard
df = pd.read_csv('data/raw/crime_tn_dataset_new.csv')

# Safety recommendations for each crime type
def get_safety_message(crime_type):
    messages = {
        'Theft': '🔒 Secure your belongings, avoid displaying valuables, and stay in well-lit areas. Keep doors and windows locked.',
        'Burglary': '🏠 Lock all doors and windows, install security systems, and avoid posting travel plans on social media. Stay safe!',
        'Cybercrime': '💻 Use strong passwords, avoid suspicious links, and never share personal information online. Enable two-factor authentication.',
        'Assault': '👥 Stay in groups, avoid isolated areas, and trust your instincts. Contact authorities if you feel threatened.',
        'Domestic Violence': '📞 Seek help immediately. Contact local authorities or helpline: 181 (Women Helpline). You are not alone.',
        'Fraud': '💳 Verify all transactions, never share OTPs or bank details, and be cautious of unsolicited calls or messages.',
        'Drug Offense': '🚫 Stay away from suspicious activities, report drug-related crimes to authorities, and seek help if needed.',
        'Traffic Violation': '🚗 Follow traffic rules, wear seatbelts, avoid speeding, and never drive under influence. Safety first!',
        'Property Crime': '🏡 Install security cameras, use proper lighting, and maintain good relationships with neighbors for community watch.',
        'Vandalism': '👀 Report suspicious activities, install surveillance, and participate in community safety programs.'
    }
    return messages.get(crime_type, '⚠️ Stay alert and follow general safety precautions. Contact local authorities if needed.')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict_form():
    districts = sorted(df['district'].unique())
    # Get taluks for each district
    district_taluks = {}
    for district in districts:
        district_taluks[district] = sorted(df[df['district'] == district]['taluk'].unique().tolist())
    print(f"Districts: {len(districts)}, Sample district_taluks: {list(district_taluks.items())[:2]}")
    return render_template('index.html', districts=districts, district_taluks=district_taluks)

@app.route('/predict', methods=['POST'])
def predict_crime():
    try:
        # Get form data
        district = request.form['district']
        taluk = request.form['taluk']
        time_of_day = request.form['time_of_day']
        day_of_week = request.form['day_of_week']
        # weather = 'Sunny'  # Removed weather
        area_type = request.form.get('area_type', 'Urban')
        age_group = request.form.get('age_group', '26-35')
        public_event = int(request.form.get('public_event', 0))
        month = int(request.form.get('month', 1))
        
        # Create feature vector with default values
        features = {
            'district_encoded': encoders['district'].transform([district])[0],
            'taluk_encoded': encoders['taluk'].transform([taluk])[0],
            'area_type_encoded': encoders['area_type'].transform([area_type])[0],
            'latitude': 11.0,  # Default Tamil Nadu center
            'longitude': 78.0,
            'month': month,
            'day_of_week_encoded': encoders['day_of_week'].transform([day_of_week])[0],
            'time_of_day_encoded': encoders['time_of_day'].transform([time_of_day])[0],
            'population_density': 500,
            'unemployment_rate': 8.0,
            'literacy_rate': 75.0,
            'poverty_index': 0.4,
            'police_station_count': 5,
            'cctv_density': 10,
            'past_crime_rate': 3.0,
            # 'weather_encoded': encoders['weather'].transform([weather])[0],  # Removed weather
            'festival_period': 1 if month in [10, 11, 12, 4] else 0,
            'age_group_encoded': encoders['age_group'].transform([age_group])[0],
            'gender_ratio': 1.0,
            'road_density': 2.0,
            'education_index': 0.75,
            'internet_penetration': 60.0,
            'alcohol_availability': 0.5,
            'transport_access': 0.7,
            'public_event': public_event,
            'is_weekend': 1 if day_of_week in ['Saturday', 'Sunday'] else 0,
            'crime_rate_index': 1.5
        }
        
        # Convert to array and scale
        feature_array = np.array(list(features.values())).reshape(1, -1)
        feature_scaled = scaler.transform(feature_array)
        
        # Predict
        prediction = model.predict(feature_scaled)[0]
        probabilities = model.predict_proba(feature_scaled)[0]
        
        # Get crime type name
        crime_type = encoders['crime_type'].inverse_transform([prediction])[0]
        confidence = max(probabilities) * 100
        
        safety_message = get_safety_message(crime_type)
        
        return render_template('result.html', 
                             crime_type=crime_type,
                             confidence=f"{confidence:.1f}",
                             district=district,
                             taluk=taluk,
                             time_of_day=time_of_day,
                             day_of_week=day_of_week,
                             age_group=age_group,
                             public_event='Yes' if public_event else 'No',
                             safety_message=safety_message)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/dashboard')
def dashboard():
    # Generate visualizations
    plots = {}
    
    # Crime distribution
    plt.figure(figsize=(10, 6))
    df['crime_type'].value_counts().plot(kind='bar')
    plt.title('Crime Type Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plots['crime_dist'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # District analysis
    plt.figure(figsize=(12, 6))
    district_crimes = df['district'].value_counts().head(10)
    district_crimes.plot(kind='bar')
    plt.title('Top 10 Districts by Crime Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plots['district_crimes'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Time patterns
    plt.figure(figsize=(10, 6))
    time_crimes = df['time_of_day'].value_counts()
    time_crimes.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Crime Distribution by Time of Day')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plots['time_pattern'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Confusion Matrix
    try:
        # Load test data
        X_test = np.load('data/processed/X_test.npy')
        y_test = np.load('data/processed/y_test.npy')
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Get crime type labels
        crime_labels = encoders['crime_type'].classes_
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=crime_labels, yticklabels=crime_labels)
        plt.title('Model Performance - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150)
        img.seek(0)
        plots['confusion_matrix'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        plots['confusion_matrix'] = None
    
    # Statistics
    stats = {
        'total_records': len(df),
        'unique_districts': df['district'].nunique(),
        'crime_types': df['crime_type'].nunique(),
        'avg_literacy': f"{df['literacy_rate'].mean():.1f}%",
        'avg_unemployment': f"{df['unemployment_rate'].mean():.1f}%"
    }
    
    return render_template('dashboard.html', plots=plots, stats=stats)

if __name__ == '__main__':
    app.run(debug=True)