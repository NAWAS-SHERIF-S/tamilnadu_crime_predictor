# Tamil Nadu Crime Pattern Predictor

A complete end-to-end AI/ML web application for predicting crime patterns in Tamil Nadu using machine learning.

## Features

- **Synthetic Dataset**: 1000 rows with 25 features including demographic, geographic, and temporal data
- **ML Pipeline**: Complete 5-phase ML lifecycle (Business Understanding → Data → Modeling → Evaluation → Deployment)
- **Web Interface**: Flask-based web application with prediction form and analytics dashboard
- **Visualizations**: Interactive charts showing crime patterns and trends

## Project Structure

```
crime_pattern_predictor/
├── data/
│   ├── raw/crime_tn_dataset.csv
│   └── processed/
├── notebooks/01_eda.ipynb
├── src/
│   ├── data_generator.py
│   ├── features/build_features.py
│   └── models/train_model.py
├── models/
├── templates/
├── reports/
├── app.py
└── requirements.txt
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate dataset and train model:
```bash
python src/data_generator.py
python src/features/build_features.py
python src/models/train_model.py
```

3. Run the web application:
```bash
python app.py
```

4. Open browser to `http://localhost:5000`

## Usage

1. **Prediction**: Enter district, time, day, and weather to predict likely crime type
2. **Dashboard**: View analytics and crime pattern visualizations
3. **Model Performance**: Check model accuracy and evaluation metrics

## Model Performance

- **Algorithm**: Logistic Regression
- **Test Accuracy**: 13.33%
- **Features**: 26 engineered features
- **Classes**: 10 crime types

## Future Improvements

- Increase dataset size for better accuracy
- Implement ensemble methods
- Add real-time data integration
- Deploy to cloud platform

## License

MIT License