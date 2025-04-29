import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# Ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]), plt.yticks([])
    plt.imshow(X_train[i])
plt.show()

# Priprema podataka
X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, dtype="uint8")
y_test = to_categorical(y_test, dtype="uint8")

# Definicija funkcija povratnog poziva
basic_callbacks = [
    keras.callbacks.TensorBoard(log_dir='logs/cnn', update_freq=100)
]

dropout_callbacks = [
    keras.callbacks.TensorBoard(log_dir='logs/cnn_dropout', update_freq=100)
]

dropout_earlystop_callbacks = [
    keras.callbacks.TensorBoard(log_dir='logs/cnn_dropout_es', update_freq=100),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
]

# 9.4.1
model1 = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(X_train_n, y_train, epochs=40, batch_size=64, callbacks=basic_callbacks, validation_split=0.1)
score1 = model1.evaluate(X_test_n, y_test, verbose=0)
print(f"Zadatak 9.4.1 - Točnost: {100.0 * score1[1]:.2f}%")

# 9.4.2
model2 = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(500, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(X_train_n, y_train, epochs=40, batch_size=64, callbacks=dropout_callbacks, validation_split=0.1)
score2 = model2.evaluate(X_test_n, y_test, verbose=0)
print(f"Zadatak 9.4.2 - Točnost s dropoutom: {100.0 * score2[1]:.2f}%")

# 9.4.3
model2.fit(X_train_n, y_train, epochs=40, batch_size=64, callbacks=dropout_earlystop_callbacks, validation_split=0.1)

# 9.4.4
print("""
Zadatak 9.4.4 - Teorijska pitanja:
1. Batch size:
   - Velika batch size: Brže treniranje, ali lošija generalizacija.
   - Mala batch size: Bolja generalizacija, ali sporije treniranje.
2. Stopa učenja:
   - Prevelika: može preskakati rješenja.
   - Premala: sporo treniranje, moguće zapinjanje u lokalnim minimumima.
3. Manja mreža (manje slojeva): manji broj parametara, brže treniranje, ali potencijalno niža točnost.
4. Manji skup za učenje: veći rizik overfittinga, model se uči na manjem uzorku podataka.
""")
