import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import argparse

from utils import preprocess_text, get_metrics


def trainval(data_path, transformer_checkpoint_path, model_checkpoint_path):
    data = pd.read_csv(data_path)
    data["text"] = data["text"].apply(preprocess_text)
    
    #train
    transformer = TfidfVectorizer(ngram_range=(1, 2), max_features=100000)
    X_train = transformer.fit_transform(data[data["split"] == "train"]["text"].values)

    model = LogisticRegression(C=5e1, solver='lbfgs', random_state=42, n_jobs=8)
    model.fit(X_train, data[data["split"] == "train"]["sentiment"])


    #validation
    X_val = transformer.transform(data[data["split"] == "val"]["text"].values)
    group_val = data[data["split"] == "val"]["source"].values 
    y_val = data[data["split"] == "val"]["sentiment"].values
    preds = model.predict(X_val)
    
    get_metrics(y_val, preds, group_val)
    
    joblib.dump(transformer, transformer_checkpoint_path)
    joblib.dump(model, model_checkpoint_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--tf_checkpoint', type=str, required=True)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    
    return parser.parse_args()    
    

if __name__ == "__main__":
    args = parse_args()
    trainval(args.data_file, args.tf_checkpoint, args.model_checkpoint)
