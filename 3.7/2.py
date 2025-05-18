import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

X = np.random.uniform(0, 10, size=(200000, 2))
y = X.sum(axis=1)

X = X / 10.0
y = y / 20.0

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    X, y,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)


def predict(a, b):
    normalized_input = np.array([[a/10.0, b/10.0]])
    prediction = model.predict(normalized_input)[0][0]
    return prediction * 20.0

test_cases = [
    (2, 2),
    (3.5, 4.5),
    (0, 10),
    (7.3, 2.7),
    (9.9, 9.9)
]

print("Результаты тестирования:")
for a, b in test_cases:
    print(f"{a} + {b} = {predict(a, b):.2f}")