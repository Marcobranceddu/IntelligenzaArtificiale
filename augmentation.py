import os
import cv2
import numpy as np
import imutils
import csv


# funzione che legge ricorsivamente tutti i file .png e .jpg nella cartella
def read_files(path):
    files = []
    for root, filenames in os.walk(path):
        for f in filenames:
            #se è png o jpg
            if f.endswith('.png') or f.endswith('.jpg'):
                files.append(os.path.join(root, f))
    return files


# funzione che legge tutte le immagini e le maschere
def read_images_and_masks(images_path, masks_path):
    images = []
    masks = []
    for image, mask in zip(images_path, masks_path):
        images.append(cv2.imread(image))
        masks.append(cv2.imread(mask))
    return images, masks


#funzione che per trovare la pendenza e l'ampiezza della maschera
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
    #l'ampiezza è quella maggiore
    amplitude = max(amplitude, amplitude2)

    return angle, amplitude

# augmentation delle immagini e delle maschere
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

print("ci sono",len(images),"immagini e",len(masks),"maschere iniaizalmente")

amplitudes, angles, rotated_images, rotated_masks, rotated_angles, mirrored_images, mirrored_masks, mirrored_angles  = rotate_and_mirror(images, masks)


#concatena le immagini e le maschere e le modifica ogni volta
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


print("Ora sono presenti ",len(augmented_images), "immagini.")
print("Ora sono presenti ",len(augmented_masks), "maschere.")


# se non esistono, crea le cartelle per le immagini e le maschere agmented
if not os.path.exists('augmented_images'):
    os.mkdir('augmented_images')
if not os.path.exists('augmented_masks'):
    os.mkdir('augmented_masks')

# salva le immagini e le maschere
for i, (image, mask) in enumerate(zip(augmented_images, augmented_masks)):
    cv2.imwrite('augmented_images/'+str(i)+'.png', image)
    cv2.imwrite('augmented_masks/'+str(i)+'.png', mask)

# crea un file csv con le ampiezze e gli angoli
with open('augmented_amplitudes.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(augmented_amplitudes)

with open('augmented_angles.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(augmented_angles)
