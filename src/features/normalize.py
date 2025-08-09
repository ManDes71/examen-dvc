import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def normalize_data(input_dir, output_dir):
    # Colonnes à exclure du scaling
    exclude_cols = ['year', 'month', 'day', 'hour']
    
    # Load train and test features
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))

    cols_to_scale = [col for col in X_train.columns if col not in exclude_cols]

    # Séparation colonnes exclues et colonnes à scaler
    X_train_excluded = X_train[exclude_cols].copy() if any(col in X_train.columns for col in exclude_cols) else pd.DataFrame()
    X_test_excluded = X_test[exclude_cols].copy() if any(col in X_test.columns for col in exclude_cols) else pd.DataFrame()

    X_train_num = X_train[cols_to_scale]
    X_test_num = X_test[cols_to_scale]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_num),columns=X_train_num.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_num),columns=X_train_num.columns)

    # Reconstruction DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=cols_to_scale)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=cols_to_scale)


    # Concaténer colonnes exclues + colonnes scalées dans l'ordre initial
    X_train_final = pd.concat([X_train_excluded, X_train_scaled_df], axis=1) if not X_train_excluded.empty else X_train_scaled_df
    X_test_final = pd.concat([X_test_excluded, X_test_scaled_df], axis=1) if not X_test_excluded.empty else X_test_scaled_df

    print(X_train_final.head(1))
    print(X_test_final.head(1))
    print("*************************")

    # Réordonner comme à l'origine
    X_train_final = X_train_final[X_train.columns]
    X_test_final = X_test_final[X_test.columns]
    print(X_train_final.head(1))
    print(X_test_final.head(1))
    print("*************************")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Save scaled datasets
    X_train_final.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    X_test_final.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)

    

if __name__ == "__main__":
    input_dir = "data/processed_data"
    output_dir = "data/processed_data"
    normalize_data(input_dir, output_dir)
