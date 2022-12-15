import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow_datasets as tfds 
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import csv


# crea una rete neurale con tensorflow
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
                metrics=['accuracy'])
    return model

#addestra la rete neurale a fornire in output l'ampiezza e l'angolo delle maschere e delle immagini associate in input
def train_model(model, train_images, train_masks, train_amplitudes, train_angles, val_images, val_masks, val_amplitudes, val_angles):
    history = model.fit(
        train_images, [train_amplitudes, train_angles],
        epochs=30,
        batch_size=32,
        validation_data=(val_images, [val_amplitudes, val_angles])
    )
    return history


#salva il modello
def save_model(model, history):
    model.save('model.h5')
    with open('history.txt', 'w') as f:
        f.write(str(history.history))
    
#carica il modello
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model
    
#addestra la rete neurale a fornire in output ampiezza e angolo delle maschere e delle immagini associate in input in augmented_masks e augmented_images
def train_augmented_model():
    model = create_model()
    train_images, train_amplitudes, train_angles = load_data()
    val_images, val_amplitudes, val_angles = load_data()
    history = train_model(model, train_images, train_amplitudes, train_angles, val_images, val_amplitudes, val_angles)
    save_model(model, history)











def load_data():
    
    images = []
    amplitudes = []
    angles = []

    #load augmented_amplitudes.csv and convert them to numpy arrays
    # Open the CSV file and read the row
    with open('augmented_amplitudes.csv') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    # Get the column that contains the strings
    string_column = rows[0]

    # Convert the strings to integers
    old_amplitudes = [int(x) for x in string_column]

    #load augmented_amplitudes.csv and convert them to numpy arrays
    # Open the CSV file and read the row
    with open('augmented_angles.csv') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]

    # Get the column that contains the strings
    string_column = rows[0]

    # Convert the strings to integers
    old_angles = [float(x) for x in string_column]


    #load images from augmented_images folder with tqdm
    for filename in tqdm(os.listdir('augmented_images_resized')):
        img = cv2.imread(os.path.join('augmented_images_resized',filename))
        #get the amplitude and the angle from the filename
        amplitude = old_amplitudes[int(filename.split('.')[0])]
        angle = old_angles[int(filename.split('.')[0])]
        if img is not None:
            
            images.append(img)
            amplitudes.append(amplitude)
            angles.append(angle)


    '''
    images = np.array(images)
    amplitudes = np.array(amplitudes)
    angles = np.array(angles)'''
    



    



    return images, amplitudes, angles

images, amplitudes, angles = load_data()

print (amplitudes, "\n\n\n\n\n\n\n\n\n", angles)

print (len(images), len(amplitudes), len(angles))