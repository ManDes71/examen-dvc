import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import yaml
import os

def load_params():
    with open(param_path, "r") as f:
        p = yaml.safe_load(f) or {}
    m = p.get("model", {})
    # valeurs par défaut sûres
    return {
        "n_estimators": m.get("n_estimators", [100, 200, 600]),
        "max_depth":   m.get("max_depth",   [None, 10, 20]),
        "min_samples_split": m.get("min_samples_split", [2, 5]),
        "min_samples_leaf":  m.get("min_samples_leaf",  [1, 2]),
        "max_features": m.get("max_features", ["sqrt", "log2"]),
        "cv": m.get("cv", 3),
        "random_state": m.get("random_state", 42),
    }

def grid_search_train(input_dir, output_dir):
    # Load scaled train data
    X_train = pd.read_csv(os.path.join(input_scaled_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv')).squeeze()
    # Remove date column if present
    if 'date' in X_train.columns or X_train.columns[0].lower().startswith('date'):
        X_train = X_train.drop(columns=[X_train.columns[0]])
    # Define model and parameter grid
    prm = load_params()
    model = RandomForestRegressor(random_state=prm["random_state"])
    param_grid = {
        "n_estimators": prm["n_estimators"],
        "max_depth": prm["max_depth"],
        "min_samples_split": prm["min_samples_split"],
        "min_samples_leaf": prm["min_samples_leaf"],
        "max_features": prm["max_features"],  
    }
    grid_search = GridSearchCV(model,  param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    # Save best parameters
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(grid_search.best_params_, os.path.join(output_dir, 'best_params.pkl'))
    print('Best parameters:', grid_search.best_params_)

if __name__ == "__main__":
    input_dir = "data/processed_data"
    input_scaled_dir = "data/processed"
    param_path= "params.yaml"
    output_dir = "models/params"
    grid_search_train(input_dir, output_dir)
