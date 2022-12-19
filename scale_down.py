import cv2
import os
from tqdm import tqdm

# funzione che ridimensiona le immagini presenti nella cartella AUGMENTED_IMAGES (256x256)
def resize_images():
    for filename in tqdm(os.listdir('augmented_images')):
        img = cv2.imread(os.path.join('augmented_images',filename))
        img = cv2.resize(img, (256,256))

        #se non esiste la cartella augmented_images_resized la crea
        if not os.path.exists('augmented_images_resized'):
            os.makedirs('augmented_images_resized')

        # salva le immagini nella nuova cartella
        cv2.imwrite(os.path.join('augmented_images_resized',filename), img)

resize_images()

'''
# funzione che ridimensiona una singola immagine .png (256x256)
def resize():
    img = cv2.imread('FILENAME.png')
    img = cv2.resize(img, (256,256))
    cv2.imwrite('chang_resized.png', img)

resize()
'''