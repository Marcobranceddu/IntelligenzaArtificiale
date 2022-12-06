
import os
import zipfile
import shutil
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import imutils

#legggi ricorsivamente tutti i file nella cartella
def read_files(path):
    files = []
    for root, dirs, filenames in os.walk(path):
        for f in filenames:
            #se è png o jpg
            if f.endswith('.png') or f.endswith('.jpg'):
                files.append(os.path.join(root, f))
    return files


#leggi tutte le immagini e le maschere
def read_images_and_masks(images_path, masks_path):
    images = []
    masks = []
    for image, mask in zip(images_path, masks_path):
        images.append(cv2.imread(image))
        masks.append(cv2.imread(mask))
    return images, masks


#trova la pendenza e l'ampiezza della maschera
def find_slope_and_amplitude(mask):
    #trova i pixel bianchi
    points = np.where(np.all(mask == [255,255,255], axis=2) == True)
    #trova top_left e bottom_right
    top_left = (points[1][0],points[0][0]) 
    bottom_right = (points[1][-1],points[0][-1])
    #trova top_right
    y_min = top_left[1]
    x_with_y_min = np.max(points[1][points[0] == y_min])
    top_right = (x_with_y_min, y_min)
    #trova bottom_left
    y_max = bottom_right[1]
    x_with_y_max = np.min(points[1][points[0] == y_max])
    bottom_left = (x_with_y_max, y_max)

    slope = (bottom_left[1]-top_left[1]) / (bottom_left[0] - top_left[0]) *-1
    #calcola angolo in gradi
    angle = np.arctan(slope) * 180 / np.pi
    if angle < 0:
        angle = 180 + angle
    #calcola ampiezza
    amplitude = top_right[0] - top_left[0]
    amplitude2 = bottom_right[0] - bottom_left[0]
    #max of amplitude and amplitude2
    amplitude = max(amplitude, amplitude2)

    return angle, amplitude

#augment le immagini e le maschere tramite albumentations
def rotate(images, masks):
    rotated_images = []
    rotated_masks = []
    for image, mask in zip(images, masks):
        angle, amplitude = find_slope_and_amplitude(mask)
        #print(angle, amplitude)
        #rotate image by random angle
        rotation_angle = np.random.randint(0,180)
        rotated_image = imutils.rotate(image, rotation_angle)
        rotated_angle = angle + rotation_angle
        if rotated_angle > 180:
            rotated_angle = rotated_angle - 180

        #show image
        print(rotated_angle)
        plt.imshow(rotated_image)
        plt.show()

        

        # transform = A.Compose([
        #     A.HorizontalFlip(p=0.5),
        #     A.RandomBrightnessContrast(p=0.2),
        #     ])
        # #ruota l'immagine di un valore casuale tra -90 e 90
        # rotate = A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=0)
        # rotated = rotate(image=image, mask=mask)
        # rotated_images.append(rotated['image'])
        # rotated_masks.append(rotated['mask'])
    return rotated_images, rotated_masks

    
images_path = read_files('zipper_aug/zipper_aug/train')
masks_path = read_files('zipper_aug/zipper_aug/masks')

images, masks = read_images_and_masks(images_path, masks_path)

rotated_images, rotated_masks = rotate(images, masks)
#concatena le immagini e le maschere
augmented_images = images + rotated_images
augmented_masks = masks + rotated_masks

print(len(augmented_images))
print(len(augmented_masks))


