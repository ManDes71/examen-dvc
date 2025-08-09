
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_and_save_model(input_dir, params_path, output_dir):
    # Load scaled train data
    X_train = pd.read_csv(os.path.join(input_scaled_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv')).squeeze()
    # Remove date column if present
    if 'date' in X_train.columns or X_train.columns[0].lower().startswith('date'):
        X_train = X_train.drop(columns=[X_train.columns[0]])
    # Load best parameters
    best_params = joblib.load(params_path)
    # Train model
    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train, y_train)
    # Save trained model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'trained_model.pkl'))
    print('Model trained and saved to', os.path.join(output_dir, 'trained_model.pkl'))

if __name__ == "__main__":
    input_dir = "data/processed_data"
    input_scaled_dir = "data/processed"
    params_path = "models/best_params.pkl"
    output_dir = "models"
    train_and_save_model(input_dir, params_path, output_dir)
