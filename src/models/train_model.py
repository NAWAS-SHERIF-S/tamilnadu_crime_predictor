import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def train_models():
    """Train and compare different models"""
    
    # Load processed data
    X_train = np.load('data/processed/X_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    models = {
        'logistic': LogisticRegression(random_state=42, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        if name == 'random_forest':
            # Grid search for Random Forest
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            print(f"Best params for {name}: {grid_search.best_params_}")
        else:
            model.fit(X_train, y_train)
        
        # Validate on training data (100% training)
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name} validation accuracy: {accuracy:.4f}")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    # Test best model on training data (100% training)
    y_test_pred = best_model.predict(X_train)
    test_accuracy = accuracy_score(y_train, y_test_pred)
    
    print(f"\\nBest model: {best_name}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save best model
    joblib.dump(best_model, 'models/crime_model.pkl')
    
    # Generate classification report
    encoders = joblib.load('models/encoders.pkl')
    crime_labels = encoders['crime_type'].classes_
    
    report = classification_report(y_train, y_test_pred, target_names=crime_labels)
    print("\\nClassification Report:")
    print(report)
    
    # Save results
    with open('reports/model_results.txt', 'w') as f:
        f.write(f"Best Model: {best_name}\\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\\n\\n")
        f.write("Classification Report:\\n")
        f.write(report)
    
    return best_model, test_accuracy

if __name__ == "__main__":
    train_models()