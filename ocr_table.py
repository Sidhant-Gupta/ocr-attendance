from cv2 import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\DELL\\ocr\\Tesseract-OCR\\tesseract.exe'

def ocr_core(img):
  text=pytesseract.image_to_string(img)
  return text


img = cv2.imread('F_01.jpg')
img=cv2.resize(img, (780, 540),  
               interpolation = cv2.INTER_NEAREST)
               
img = cv2.fastNlMeansDenoisingColored(img,None,None,None,None,None)

#converting to grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)

#removing noise from binary image
adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
cv2.imshow('adaptive_threshold',adaptive_threshold)

#edges
edges = cv2.Canny(adaptive_threshold,50,150,apertureSize = 3)
cv2.imshow('edges', edges)
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=50,maxLineGap=120)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('image', img)

# img = cv2.resize(img, None, fx=0.5, fy=0.5)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
print(ocr_core(img)) 



k = cv2.waitKey(0)
cv2.destroyAllWindows()