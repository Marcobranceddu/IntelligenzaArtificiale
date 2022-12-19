import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_mask(image, dim, amp, ang):

    # trasforma gli angoli da gradi a radianti
    ang = ang * np.pi / 180

    # crea un'immagine nera (maschera) della stessa dimensione di image
    mask = np.zeros((dim, dim, 3), np.uint8)    

    top_y = 0
    bottom_y = dim-1

    # calcola le coordinate della linea che passa per il centro dell'immagine ed ha angolo = ang
    x = (dim/2) * (np.tan(np.pi/2 - ang))
    top_x = dim/2 + x
    bottom_x = dim/2 - x

    # trova i punti del poligono da applicare sulla maschera
    top_sx = (top_x - amp*dim/2, top_y)
    top_dx = (top_x + amp*dim/2, top_y)
    bottom_sx = (bottom_x - amp*dim/2, bottom_y)
    bottom_dx = (bottom_x + amp*dim/2, bottom_y)

    # crea un poligono con i vertici trovati sopra
    pts = np.array([top_sx, top_dx, bottom_dx, bottom_sx], np.int32)
    pts = pts.reshape((-1,1,2))

    # rende il poligono bianco
    cv2.fillPoly(mask, [pts], (255,255,255))

    # stampa la maschera creata
    plt.imshow(mask)
    plt.show()

    # applica la maschera all'immagine (bitwise AND)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image