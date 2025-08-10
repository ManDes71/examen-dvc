import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def evaluate_model(input_dir, model_path, metrics_path):
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv')).squeeze()
    # Remove date column if present
    if 'date' in X_test.columns or X_test.columns[0].lower().startswith('date'):
        X_test = X_test.drop(columns=[X_test.columns[0]])
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({'mse': mse, 'r2': r2}, f)
    print('Evaluation metrics saved to', metrics_path)

if __name__ == "__main__":
    input_dir = "data/processed_data"
    model_path = "models/trained_model.pkl"
    metrics_path = "metrics/scores.json"
    evaluate_model(input_dir, model_path, metrics_path)
