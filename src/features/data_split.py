import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def load_params():
    with open(param_path, "r") as f:
        p = yaml.safe_load(f)
    test_size = float(p.get("split", {}).get("test_size", 0.2))
    random_state = int(p.get("split", {}).get("random_state", 42))
    return test_size, random_state

def split_raw_data(input_path, output_dir):
    # Load the raw data
    df = pd.read_csv(input_path)


    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df = df.drop(columns=['date'])

    print(df.head())

    target_col = "silica_concentrate"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Split into train and test sets
    test_size, random_state = load_params()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Save datasets
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

if __name__ == "__main__":
    input_path = "data/raw_data/raw.csv"
    param_path= "params.yaml"
    output_dir = "data/processed_data"
    split_raw_data(input_path, output_dir)
