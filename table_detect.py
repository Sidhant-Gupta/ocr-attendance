import cv2
import numpy as np
import matplotlib.pyplot as plt


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img1=cv2.imread('scanned_filled.jpeg')
img2=cv2.imread('scanned_empty.jpeg')
img1 = cv2.resize(img1, None, fx=0.5, fy=0.5)
img2= cv2.resize(img2, None, fx=0.5, fy=0.5)
img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

result_image = cv2.subtract(img2, img1)
plotting = plt.imshow(result_image,cmap='gray')
plt.show()
cv2.imshow('Final Image', result_image)
