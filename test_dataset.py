# prova.py  (da eseguire in gesture_challenge/)
import numpy as np
import tensorflow as tf
import data, evaluate

ROOT = "../datasets"                       # cartella con DVSGestureROI_fs32/
train, test = data.load_dataset(root=ROOT)    # frame da 200 ms (default)
train["roi"].shape       # (N, 7, 32, 32, 2)   frame da 200 ms
train["pos_map"].shape   # (N, 7, 32, 32, 1)
train["pos_events"][0]   # eventi di posizione della prima registrazione
train["y"].shape         # (N,)

# validation: split per registrazione (mai frame della stessa registrazione in train e val)
tr, va = data.train_val_split(train)
# un frame = un esempio: ROI + pos_map come canale in più -> [32, 32, 3]
def frames(split):
    x = np.concatenate([split["roi"], split["pos_map"]], axis=-1)   # [N, T, 32, 32, 3]
    N, T = x.shape[:2]
    return x.reshape(N * T, 32, 32, 3), np.repeat(split["y"], T)

x_tr, y_tr = frames(tr)
x_val, y_val = frames(va)

model = tf.keras.Sequential([
    tf.keras.layers.Input((32, 32, 3)),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(11, activation="softmax"),
])
print("parametri:", model.count_params())
model.summary()
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=30, batch_size=128)

# test: probabilità per frame [N, T, 11] -> accuracy per registrazione
x_te, _ = frames(test)
print("Load test dataset:", x_te.shape, test["y"].shape)
probs = model.predict(x_te, verbose=0).reshape(len(test["y"]), -1, 11)
print("TEST accuracy:", evaluate.accuracy(probs, test["y"]))