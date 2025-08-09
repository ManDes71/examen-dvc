import pandas as pd
import joblib
import os

def predict_and_save(input_dir, model_path, output_path):
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test_scaled.csv'))
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
    predictions.to_csv(output_path, index=False)
    print('Predictions saved to', output_path)

if __name__ == "__main__":
    input_dir = "data/processed_data"
    model_path = "models/trained_model.pkl"
    output_path = "data/predictions.csv"
    predict_and_save(input_dir, model_path, output_path)
