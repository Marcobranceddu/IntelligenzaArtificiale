import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from mask_on import apply_mask


img_to_predict = cv2.imread('422.png')

model = tf.keras.models.load_model('model_3.h5')
prediction = model.predict(np.array([img_to_predict]))

print("I valori predetti di AMPIEZZA e ANGOLO per l'immagine passata sono: ",prediction)

masked_image = apply_mask(img_to_predict, img_to_predict.shape[0], float(prediction[0][0])/1024, float(prediction[0][1]))

# stampa maschera prodotta applicata all'immagine
plt.imshow(masked_image)
plt.show()