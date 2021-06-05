import time
import json
import datetime
import tensorflow as tf
from pathlib import Path
from data.preprocessing import prepare_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from sklearn.model_selection import train_test_split


gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if gpu_devices: tf.config.experimental.set_memory_growth(gpu_devices[0], True)

INPUT_LENGTH = 500
OUTPUT_DIM = 32
INPUT_UNITS = 100
LEARNING_RATE = 0.01
EPOCHS = 5
BATCH_SIZE = 64

start_time = int(time.time())
tokenizer_path = f"models/tokenizer_{start_time}.pickle"

all_encoded_texts, all_labels, INPUT_DIM = prepare_data(tokenizer_path)
X_train, X_test, y_train, y_test = train_test_split(
    all_encoded_texts,
    all_labels,
    test_size=0.2
    )

model = Sequential()
model.add(Embedding(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, input_length=INPUT_LENGTH))
model.add(LSTM(INPUT_UNITS))
model.add(Dense(6, activation="sigmoid"))

model.compile(
    loss="categorical_crossentropy", 
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy"]
    )

print(model.summary())


if __name__ == "__main__":
    model_name = f"models/mvnews_classifier_{start_time}.h5"
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    model.save(model_name)
    
    checkpoint_path = "models/checkpoints.json"
    checkpoint_data = {
                "datetime": str(datetime.datetime.today()),
                "model_path": model_name,
                "tokenizer_path": tokenizer_path,
                "hparams": {
                    "input_dim": INPUT_DIM,
                    "input_length": INPUT_LENGTH,
                    "output_dim": OUTPUT_DIM,
                    "input_units": INPUT_UNITS,
                    "leaning_rate": LEARNING_RATE,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE
                },
                "history": {
                    "loss": history.history["loss"][-1],
                    "accuracy": history.history["accuracy"][-1],
                    "val_loss": history.history["val_loss"][-1],
                    "val_accuracy": history.history["val_accuracy"][-1]
                }
            }

    if not Path(checkpoint_path).is_file():
        with open(checkpoint_path, "w") as f:
            json.dump({"models": [checkpoint_data]}, f, indent=4)
    else:
        with open(checkpoint_path,'r+') as f:
            file_data = json.load(f)
            file_data["models"].append(checkpoint_data)
            f.seek(0)
            json.dump(file_data, f, indent=4)
            f.truncate()