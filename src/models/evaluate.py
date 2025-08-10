import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os

def evaluate_model(input_dir, input_scaled_dir, model_path, metrics_path):
    X_test = pd.read_csv(os.path.join(input_scaled_dir, 'X_test_scaled.csv'))
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

def predict_and_save(input_scaled_dir, model_path, output_dir):
    X_test = pd.read_csv(os.path.join(input_scaled_dir, 'X_test_scaled.csv'))
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
   # Reconstituer la date
    if all(col in X_test.columns for col in ['year', 'month', 'day', 'hour']):
        dates = pd.to_datetime(X_test[['year', 'month', 'day', 'hour']])
    else:
        dates = pd.Series([None] * len(X_test))
    # Sauvegarder avec la date et la prédiction
    predictions = pd.DataFrame({
        'date': dates,
        'prediction': y_pred
    })
    os.makedirs(output_dir, exist_ok=True)
    predictions.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print('Predictions saved to', os.path.join(output_dir, 'predictions.csv'))

if __name__ == "__main__":
    input_dir = "data/processed_data"
    input_scaled_dir = "data/processed"
    model_path = "models/models/trained_model.pkl"
    output_dir = "data/predictions"
    metrics_path = "metrics/scores.json"
    evaluate_model(input_dir, input_scaled_dir, model_path, metrics_path)
    predict_and_save(input_scaled_dir, model_path, output_dir)
