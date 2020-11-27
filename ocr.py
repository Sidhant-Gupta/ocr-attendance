import cv2
import numpy as numpy
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_core(img):
  text=pytesseract.image_to_string(img)
  return text

img=cv2.imread('F_PA3.jpg')

# 2. Resize the image
img = cv2.resize(img, None, fx=0.5, fy=0.5)
# 3. Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 4. Convert image to black and white (using adaptive threshold)
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)

# def getGrayScale(image):
#   cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 

# def remNoise(image):
#   return cv2.medianBlur(image,5)

# def thresholding(image):
#   return cv2.threshold(image,0,255,cv2.THRESH_BINARY + cv2.THRES_OTSU)[1]

# img=getGrayScale(img)
# img=thresholding(img)
# img=remNoise(img)

print(ocr_core(img)) 