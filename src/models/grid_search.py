import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os


def grid_search_train(input_dir, output_dir):
    # Load scaled train data
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv')).squeeze()
    # Remove date column if present
    if 'date' in X_train.columns or X_train.columns[0].lower().startswith('date'):
        X_train = X_train.drop(columns=[X_train.columns[0]])
    # Define model and parameter grid
    model = RandomForestRegressor(random_state=42)
    param_grid = {
    'n_estimators': [ 300, 500, 600],
    'max_depth': [None,  20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2 ],
    'max_features': ['auto', 'sqrt']
    }
    grid_search = GridSearchCV(model,  param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    # Save best parameters
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(grid_search.best_params_, os.path.join(output_dir, 'best_params.pkl'))
    print('Best parameters:', grid_search.best_params_)

if __name__ == "__main__":
    input_dir = "data/processed_data"
    output_dir = "models"
    grid_search_train(input_dir, output_dir)
