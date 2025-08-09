import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===== Utils chargement tolérant (PKL/JSON) =====
def load_pkl_or_json(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # tente les deux
    base, _ = os.path.splitext(path)
    for cand in (base + ".pkl", base + ".json"):
        if os.path.exists(cand):
            return load_pkl_or_json(cand)
    raise FileNotFoundError(f"Fichier introuvable (ni .pkl ni .json): {path}")

def ensure_mapper_is_dict(m):
    # mapper peut être {code_texte: id_int} (comme produit par DataImporter) ou l’inverse
    # On veut un dict {str(class_index)->prdtypecode_original}
    # Si les clés ne sont pas des str d’indices, on retourne tel quel.
    return m

# ===== Prétraitements =====
def preprocess_text_series(tokenizer, texts, maxlen=10):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")

def preprocess_image_path(image_path, target_size=(224, 224, 3)):
    img = load_img(image_path, target_size=target_size[:2])
    arr = img_to_array(img)
    return preprocess_input(arr)

def build_image_tensor(paths):
    arrs = [preprocess_image_path(p) for p in paths]
    return tf.convert_to_tensor(arrs, dtype=tf.float32)

# ===== Chargement des modèles =====
def load_models_and_assets(
    tokenizer_path="models/tokenizer_config.json",
    lstm_path="models/best_lstm_model.h5",
    vgg_path="models/best_vgg16_model.h5",
    weights_path="models/best_weights",  # sans extension: gère .pkl ou .json
    mapper_path="models/mapper",          # idem
):
    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_cfg = f.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tok_cfg)

    lstm = keras.models.load_model(lstm_path)
    vgg = keras.models.load_model(vgg_path)

    best_weights = load_pkl_or_json(weights_path)  # (w_lstm, w_vgg)
    if isinstance(best_weights, dict) and "0" in best_weights:
        # au cas où ça viendrait d’un json avec clés "0","1"
        best_weights = (float(best_weights["0"]), float(best_weights["1"]))

    mapper = load_pkl_or_json(mapper_path)
    mapper = ensure_mapper_is_dict(mapper)

    return tokenizer, lstm, vgg, best_weights, mapper

# ===== Prédiction =====
def run_predict(dataset_path, images_path, out_path):
    # Chargements
    tokenizer, lstm, vgg, (w_lstm, w_vgg), mapper = load_models_and_assets()

    # Lecture des 10 premières lignes comme dans predict.py d’origine
    X = pd.read_csv(dataset_path).head(10)

    # Recompose le chemin image à la manière de ImagePreprocessor:contentReference[oaicite:5]{index=5}
    if "image_path" not in X.columns and {"imageid","productid"}.issubset(X.columns):
        X["image_path"] = (
            f"{images_path}/image_" + X["imageid"].astype(str)
            + "_product_" + X["productid"].astype(str) + ".jpg"
        )

    # Texte -> séquences
    padded = preprocess_text_series(tokenizer, X["description"].fillna(""), maxlen=10)

    # Images -> tenseur
    imgs = build_image_tensor(X["image_path"])

    # Probas
    proba_lstm = lstm.predict([padded], verbose=0)
    proba_vgg  = vgg.predict([imgs],    verbose=0)

    proba = w_lstm * proba_lstm + w_vgg * proba_vgg
    y_pred_idx = np.argmax(proba, axis=1)

    # Si mapper est {prdtypecode: index}, on veut l’inverse pour retrouver le code original
    if all(isinstance(v, (int, np.integer)) for v in mapper.values()):
        inv = {int(v): k for k, v in mapper.items()}
        decode = lambda i: inv.get(int(i), int(i))
    else:
        # sinon, assume déjà forme index->label
        decode = lambda i: mapper.get(str(int(i)), int(i))

    predictions = {int(i): decode(idx) for i, idx in enumerate(y_pred_idx)}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"Prédictions enregistrées dans : {out_path}")

# ===== Évaluation =====
def run_evaluate(dataset_path, images_path, y_path=None, id_cols=("productid","imageid"), out_path="data/preprocessed/metrics.json"):
    """
    - Si dataset_path contient déjà une colonne 'prdtypecode', on l'utilise.
    - Sinon, on charge y_path (CSV) et on fusionne sur id_cols (par défaut productid + imageid).
    """
    tokenizer, lstm, vgg, (w_lstm, w_vgg), mapper = load_models_and_assets()

    X = pd.read_csv(dataset_path)
    if "prdtypecode" in X.columns:
        y_true = X["prdtypecode"].values
    else:
        if y_path is None:
            raise ValueError("Aucun label trouvé. Fournis --y_path ou un CSV avec 'prdtypecode'.")
        Y = pd.read_csv(y_path)
        # tentative de merge sur colonnes communes (id produits / images)
        merge_keys = [c for c in id_cols if c in X.columns and c in Y.columns]
        if not merge_keys:
            raise ValueError("Impossible de fusionner X et Y: clés communes manquantes.")
        X = X.merge(Y, on=merge_keys, how="inner")
        y_true = X["prdtypecode"].values

    # Chemins images
    if "image_path" not in X.columns and {"imageid","productid"}.issubset(X.columns):
        X["image_path"] = (
            f"{images_path}/image_" + X["imageid"].astype(str)
            + "_product_" + X["productid"].astype(str) + ".jpg"
        )

    padded = preprocess_text_series(tokenizer, X["description"].fillna(""), maxlen=10)
    imgs = build_image_tensor(X["image_path"])

    proba_lstm = lstm.predict([padded], verbose=0)
    proba_vgg  = vgg.predict([imgs],    verbose=0)
    proba = w_lstm * proba_lstm + w_vgg * proba_vgg
    y_pred_idx = np.argmax(proba, axis=1)

    # Si y_true est encodé 0..26 (comme dans DataImporter:contentReference[oaicite:6]{index=6}), on peut évaluer direct
    acc = float(accuracy_score(y_true, y_pred_idx))
    report = classification_report(y_true, y_pred_idx, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred_idx).tolist()

    metrics = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Metrics enregistrées dans : {out_path}")

# ===== CLI =====
def main():
    parser = argparse.ArgumentParser(description="Prédire ou évaluer un modèle (texte+image).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="Générer des prédictions.")
    p_pred.add_argument("--dataset_path", default="data/preprocessed/X_train_update.csv", type=str)
    p_pred.add_argument("--images_path",  default="data/preprocessed/image_train", type=str)
    p_pred.add_argument("--out_path",     default="data/preprocessed/predictions.json", type=str)

    p_eval = sub.add_parser("evaluate", help="Évaluer le modèle.")
    p_eval.add_argument("--dataset_path", default="data/preprocessed/X_train_update.csv", type=str)
    p_eval.add_argument("--images_path",  default="data/preprocessed/image_train", type=str)
    p_eval.add_argument("--y_path",       default=None, type=str, help="CSV des labels si X ne contient pas prdtypecode")
    p_eval.add_argument("--out_path",     default="data/preprocessed/metrics.json", type=str)

    args = parser.parse_args()

    if args.cmd == "predict":
        run_predict(args.dataset_path, args.images_path, args.out_path)
    else:
        run_evaluate(args.dataset_path, args.images_path, args.y_path, out_path=args.out_path)

if __name__ == "__main__":
    main()

