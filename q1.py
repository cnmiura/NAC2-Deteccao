#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt


temp = cv2.imread('img.png', 0) # carta maior
temp2 = cv2.imread('img2.png', 0) # carta menor

face_width, face_height = temp.shape[::-1]
face_width2, face_height2 = temp2.shape[::-1]


threshold = 0.4


cap = cv2.VideoCapture("q1.mp4")


while True:
    ret, frame = cap.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    # Seu código aqui. 
    res = cv2.matchTemplate(img_gray, temp,cv2.TM_CCOEFF_NORMED)
    res2 = cv2.matchTemplate(img_gray,temp2,cv2.TM_CCOEFF_NORMED)

    # carta maior
    location = np.where(res >= threshold)
    for pt in zip(*location[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + face_width, pt[1] + face_height), (0, 255, 0), 2)
        cv2.putText(frame, "CARTA DETECTADA", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
    # carta menor
    location2 = np.where(res2 >= threshold)
    for pt2 in zip(*location2[::-1]):
        cv2.rectangle(frame, pt2, (pt2[0] + face_width2, pt2[1] + face_height2), (0, 255, 0), 2)
        cv2.putText(frame, "CARTA DETECTADA", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Exibe resultado
    cv2.imshow("Frame", frame)

    #if not ret:
    #    break

    # Wait for key 'ESC' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# That's how you exit
cap.release()
cv2.destroyAllWindows()