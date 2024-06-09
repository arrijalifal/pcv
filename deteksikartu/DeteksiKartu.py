# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 05:59:22 2023

@author: visikom
"""

import cv2
import numpy as np


def apply_green_filter(image_path):
    # Membaca gambar
    image = cv2.imread(image_path)
    
    # Mengkonversi gambar ke ruang warna HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mendefinisikan rentang warna hijau dalam HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Membuat mask untuk warna hijau
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_not(maskg)
    kernel = np.ones((5,5), np.uint8)
    mask  = cv2.erode(mask, kernel, iterations=1)  
    mask  = cv2.dilate(mask, kernel, iterations=1)  
    
    num_labels, labels = cv2.connectedComponents(mask)
    print(num_labels)
    # Menggabungkan mask dengan gambar asli
    l =[] 
    print(np.max(labels))
    for i in range(num_labels):
        b,c = np.where(labels==i)
        bmin =np.min(b) 
        bmax =np.max(b)
        cmin =np.min(c) 
        cmax =np.max(c)
        im = image[bmin:bmax ,cmin:cmax ,:]
        l.append(im)
      #  cv2.imshow(str(i),im)
        luas =( bmax - bmin )*(cmax-cmin)
        print(i," Luas ",luas )
        
        
    green_filter = cv2.bitwise_and(image, image, mask=mask)

    # Menampilkan gambar asli dan hasil filter
    cv2.imshow('Original Image', image)
    cv2.imshow('mask', mask)
    cv2.imshow('maskG', maskg)
    
    
    cv2.imshow('Green Filter Applied', green_filter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ganti 'path_to_image.jpg' dengan path ke gambar Anda
apply_green_filter('deteksikartu/kartu.png')
