import time
import pickle
import tensorflow as tf
from kerastuner.tuners import RandomSearch
from data.preprocessing import prepare_data
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, Embedding


def build_model(hp):
    output_dim = hp.Int("output_dim", min_value=32, max_value=512, step=16)
    input_unit = hp.Int("input_unit", min_value=32, max_value=512, step=16)
    
    model = Sequential()
    model.add(Embedding(input_dim=260383, output_dim=output_dim, input_length=500))
    model.add(LSTM(input_unit))
    model.add(Dense(5, activation='sigmoid'))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  metrics=['accuracy'])
    return model

all_encoded_texts, all_labels = prepare_data(tokenizer_path=None)
X_train, X_test, y_train, y_test = train_test_split(
    all_encoded_texts,
    all_labels,
    test_size=0.2
    )


if __name__ == "__main__":
    tuner = RandomSearch(
        build_model,
        objective = "val_accuracy",
        max_trials = 20,
        executions_per_trial = 3,
        directory = f"tuners/trial_model_{int(time.time())}",
        project_name = "news_classifier"
    )

    tuner.search(
        x = X_train,
        y = y_train,
        epochs = 6,
        batch_size = 64,
        validation_data = (X_test, y_test)
    )
    
    with open(f"tuner_main{int(time.time())}.pkl", "wb") as f:
        pickle.dump(tuner, f)
