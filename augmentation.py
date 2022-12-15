import os
import zipfile
import shutil
import cv2
import numpy as np
import albumentations as A

import matplotlib.pyplot as plt
import imutils
import csv




    


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
def rotate_and_mirror(images, masks):
    rotated_images = []
    rotated_masks = []
    images_amplitude = []
    images_angle = []
    rotated_images_angles = []
    mirrored_images = []
    mirrored_masks = []
    mirrored_images_angles = []

    for i in range(len(images)):
        #trova angolo e ampiezza
        angle, amplitude = find_slope_and_amplitude(masks[i])
        rotation_angle = np.random.randint(0,180)
        rotated_image = imutils.rotate(images[i], rotation_angle)
        rotated_mask = imutils.rotate(masks[i], rotation_angle)
        rotated_angle = angle + rotation_angle
        if rotated_angle > 180:
            rotated_angle = rotated_angle - 180

        mirrored_image = cv2.flip(rotated_image, 0)
        mirrored_mask = cv2.flip(rotated_mask, 0)
        mirrored_angle = 180 - rotated_angle


        images_amplitude.append(amplitude)
        images_angle.append(angle)

        rotated_images.append(rotated_image)
        rotated_masks.append(rotated_mask)
        rotated_images_angles.append(rotated_angle)

        mirrored_images.append(mirrored_image)
        mirrored_masks.append(mirrored_mask)
        mirrored_images_angles.append(mirrored_angle)
        
    '''
    for image, mask in zip(images, masks):
        angle, amplitude = find_slope_and_amplitude(mask)
        #print(angle, amplitude)
        #rotate image by random angle between 0° and 180°
        rotation_angle = np.random.randint(0,180)
        rotated_image = imutils.rotate(image, rotation_angle)
        rotated_angle = angle + rotation_angle
        if rotated_angle > 180:
            rotated_angle = rotated_angle - 180

        #show image
        #print("L'angolo di questa immagine è:",angle)
        #print("L'ampiezza di questa immagine è:",amplitude)

        
        plt.imshow(image)
        plt.show()

        print("L'angolo di questa immagine RUOTATA è:",rotated_angle)
        print("L'ampiezza di questa immagine è:",amplitude)

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
        '''
    return images_amplitude, images_angle, rotated_images, rotated_masks, rotated_images_angles, mirrored_images, mirrored_masks, mirrored_images_angles


def sfoca(aug_imgs):
    blurreds = []
    for img in aug_imgs:
        blurreds.append(cv2.GaussianBlur(img,(45,45),0))
    return blurreds

def change_luminosity(aug_imgs):
    changed = []
    for img in aug_imgs:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v += 50
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        changed.append(img)
    return changed

def change_contrast(aug_imgs):
    changed = []
    for img in aug_imgs:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.where((255-v)<50,255,v+50)
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        changed.append(img)
    return changed
    
images_path = read_files('zipper_aug/zipper_aug/train')
masks_path = read_files('zipper_aug/zipper_aug/masks')

images, masks = read_images_and_masks(images_path, masks_path)

print("ci sono",len(images),"immagini e",len(masks),"maschere ORIGINALI")

amplitudes, angles, rotated_images, rotated_masks, rotated_angles, mirrored_images, mirrored_masks, mirrored_angles  = rotate_and_mirror(images, masks)

print("ci sono",len(rotated_images),"immagini e",len(rotated_masks),"maschere RUOTATE")

print("ci sono",len(mirrored_images),"immagini e",len(mirrored_masks),"maschere SPECCHIATE")

#concatena le immagini e le maschere
augmented_images = images + rotated_images + mirrored_images
augmented_masks = masks + rotated_masks + mirrored_masks
augmented_amplitudes = amplitudes + amplitudes + amplitudes
augmented_angles = angles + rotated_angles + mirrored_angles

blurred_images = sfoca(augmented_images)
augmented_images = augmented_images + blurred_images
augmented_masks = augmented_masks + augmented_masks
augmented_amplitudes = augmented_amplitudes + augmented_amplitudes
augmented_angles = augmented_angles + augmented_angles

changed_luminosity_images = change_luminosity(augmented_images)
augmented_images = augmented_images + changed_luminosity_images
augmented_masks = augmented_masks + augmented_masks
augmented_amplitudes = augmented_amplitudes + augmented_amplitudes
augmented_angles = augmented_angles + augmented_angles

changed_contrast_images = change_contrast(augmented_images)
augmented_images = augmented_images + changed_contrast_images
augmented_masks = augmented_masks + augmented_masks
augmented_amplitudes = augmented_amplitudes + augmented_amplitudes
augmented_angles = augmented_angles + augmented_angles






print(len(augmented_images))
print(len(augmented_masks))


print("\n\n\n\n\n\n\n")
print(augmented_amplitudes[16], augmented_amplitudes[416], augmented_amplitudes[816], augmented_amplitudes[1216], augmented_amplitudes[1616], augmented_amplitudes[2016])
print(augmented_angles[16], augmented_angles[416], augmented_angles[816], augmented_angles[1216], augmented_angles[1616], augmented_angles[2016])

#crea una cartella e mettici dentro augmented_images e augmented_masks
if not os.path.exists('augmented_images'):
    os.mkdir('augmented_images')
if not os.path.exists('augmented_masks'):
    os.mkdir('augmented_masks')

#salva le immagini e le maschere
for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks)):
    cv2.imwrite('augmented_images/'+str(i)+'.png', image)
    cv2.imwrite('augmented_masks/'+str(i)+'.png', mask)

#crea un file csv con le ampiezze e gli angoli
with open('augmented_amplitudes.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(augmented_amplitudes)

with open('augmented_angles.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(augmented_angles)



print("AIO' FINI' BELOOOOOOOO")
