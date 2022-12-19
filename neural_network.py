import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import csv
from sklearn.model_selection import train_test_split


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

# funzione che legge e carica i vari dati (immagini, ampiezze e angoli)
def load_data():
    
    images = []
    amplitudes = []
    angles = []

    # apre il file .csv e legge le righe (ampiezze)
    with open('augmented_amplitudes.csv') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    string_column = rows[0]

    old_amplitudes = [int(x) for x in string_column]

    # apre il file .csv e legge le righe (angoli)
    with open('augmented_angles.csv') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    string_column = rows[0]

    old_angles = [float(x) for x in string_column]

    # legge e carica le immagi
    for filename in tqdm(os.listdir('augmented_images_resized')):
        img = cv2.imread(os.path.join('augmented_images_resized',filename))

        # usa il nome del file per recuperare l'ampiezza e l'angolo corrispondente
        amplitude = old_amplitudes[int(filename.split('.')[0])]
        angle = old_angles[int(filename.split('.')[0])]

        if img is not None:
            images.append(img)
            amplitudes.append(amplitude)
            angles.append(angle)

    return images, amplitudes, angles


images, amplitudes, angles = load_data()

x = np.array(images)
angles = np.array(angles)
amplitudes = np.array(amplitudes)

y = np.stack((amplitudes, angles), axis=1)

# suddivide il dataset in train e validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

batch_size = 100
print("batch_size: "+str(batch_size))

#early stopping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)


model_1 = create_model()

run_hist_1=model_1.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=100,
              validation_data=(x_val, y_val),
              callbacks=[callback],
              shuffle=True)

print("terminato a epoca: ", len(run_hist_1.history['loss']))

model_1.save('model_1.h5')