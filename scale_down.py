import cv2
import os
from tqdm import tqdm

#reduce the resolution of every image in the folder "augmented_images" and save them in the folder "augmented_images_resized"
def resize_images():
    for filename in tqdm(os.listdir('augmented_images')):
        img = cv2.imread(os.path.join('augmented_images',filename))
        img = cv2.resize(img, (256,256))

        #se non esiste la cartella augmented_images_resized la crea
        if not os.path.exists('augmented_images_resized'):
            os.makedirs('augmented_images_resized')

        cv2.imwrite(os.path.join('augmented_images_resized',filename), img)
resize_images()

