import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from data.helpers import cleaner
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


dataset_path = "data/dataset.csv"
def prepare_data(tokenizer_path):
    if Path(dataset_path).is_file():
        news = pd.read_csv(dataset_path, encoding="utf8")
    else:
        print("data\dataset.csv not found.")
        quit()
    
    class_0 = news[news["category"] == "politics"]
    class_1 = news[news["category"] == "viyafaari"]
    class_2 = news[news["category"] == "sport"]
    class_3 = news[news["category"] == "world-news"]
    class_4 = news[news["category"] == "report"]

    all_texts = np.append(class_0["title_body"], class_1["title_body"])
    all_texts = np.append(all_texts, class_2["title_body"])
    all_texts = np.append(all_texts, class_3["title_body"])
    all_texts = np.append(all_texts, class_4["title_body"])

    all_cleaned_texts = np.array([cleaner(text) for text in all_texts])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_cleaned_texts)
    t = tokenizer.word_index
    print("word_index:", len(t))

    if tokenizer_path:
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_encoded_texts = tokenizer.texts_to_sequences(all_cleaned_texts)
    all_encoded_texts = np.array(all_encoded_texts)
    all_encoded_texts = sequence.pad_sequences(all_encoded_texts, maxlen=500) #, maxlen=40

    labels_0 = np.array([0] * len(class_0))
    labels_1 = np.array([1] * len(class_1))
    labels_2 = np.array([2] * len(class_2))
    labels_3 = np.array([3] * len(class_3))
    labels_4 = np.array([4] * len(class_4))

    all_labels = np.append(labels_0, labels_1)
    all_labels = np.append(all_labels, labels_2)
    all_labels = np.append(all_labels, labels_3)
    all_labels = np.append(all_labels, labels_4)

    all_labels = all_labels[:, np.newaxis]

    one_hot_encoder = OneHotEncoder(sparse=False)
    all_labels = one_hot_encoder.fit_transform(all_labels)

    return (all_encoded_texts, all_labels, len(t)+1)