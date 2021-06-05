import json
import pickle
import numpy as np
import tensorflow as tf
from data.helpers import cleaner
from tensorflow.keras.preprocessing import sequence


def start_model(model_path, tokenizer_path):
    print(f"\nModel Path: {model_path}")
    print(f"Tokenizer Path: {tokenizer_path}\n")
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    print("\n")

    return (model, tokenizer)

categories = ["politics", "business", "sports", "world-news", "report"]
def preditct_news_topic(text):
    cleaned_text = cleaner(text)
    encoded_text = tokenizer.texts_to_sequences([cleaned_text])
    encoded_text = sequence.pad_sequences(encoded_text, maxlen=500)
    predict_topic = model.predict(encoded_text)
    prediction = np.argmax(predict_topic)
    return prediction


if __name__ == "__main__":
    import argparse
    import pandas as pd
    from pprint import pprint

    parser = argparse.ArgumentParser(description="Select which model to load for inference.")
    parser.add_argument("-b", "--best", action="store_true", help="load the model with the best accuracy.")
    parser.add_argument("-m", "--model", action="store_true", help="select which model to load.")
    args = parser.parse_args()

    checkpoint_path = "models/checkpoints.json"
    try:
        with open(checkpoint_path, "r") as f:
            checkpoints = json.load(f)
    except FileNotFoundError:
        print(checkpoint_path, "not found. Make sure you train the model before runing inference.py")
        quit()

    if args.best:
        best_model = sorted(checkpoints["models"], key=lambda k: k["history"]["val_accuracy"], reverse=True)[0]
        tokenizer_path = best_model["tokenizer_path"]
        model_path =  best_model["model_path"]
        pprint({"hparams": best_model["hparams"]})
        pprint({"history": best_model["history"]})

    elif args.model:
        models = sorted(checkpoints["models"], key=lambda k: k["history"]["val_accuracy"], reverse=True)
        for x in models:
            x["hparams"]["accuracy"] = x["history"]["val_accuracy"]
        df = pd.DataFrame([x["hparams"] for x in models])
        print(df.to_string())
        select_model = input("Select a model to load: ")
        tokenizer_path = models[int(select_model)]["tokenizer_path"]
        model_path = models[int(select_model)]["model_path"]

    else:
        latest_model = checkpoints["models"][-1]
        tokenizer_path = latest_model["tokenizer_path"]
        model_path = latest_model["model_path"]
        pprint({"hparams": latest_model["hparams"]})
        pprint({"history": latest_model["history"]})

    model, tokenizer = start_model(model_path, tokenizer_path)

    while True:
        text = input("Article: ")
        p = preditct_news_topic(text)
        print(categories[p])
