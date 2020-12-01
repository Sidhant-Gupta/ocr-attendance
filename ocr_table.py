import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_core(img):
  text=pytesseract.image_to_string(img)
  return text

blankImg=cv2.imread('blank.png')
blankImg=cv2.resize(blankImg, (780, 540),  
               interpolation = cv2.INTER_NEAREST)
img = cv2.imread('F_01.jpg')
img=cv2.resize(img, (780, 540),  
               interpolation = cv2.INTER_NEAREST) 

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow('edges', edges)
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.line(blankImg,(x1,y1),(x2,y2),(0,255,0),2)

img1=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(blankImg, cv2.COLOR_BGR2GRAY)

result_image = cv2.subtract(blankImg, img)
cv2.imshow('subtractimage', result_image)
# cv2.imshow('image', img)
cv2.imshow('image', blankImg)

# img = cv2.resize(img, None, fx=0.5, fy=0.5)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
print(ocr_core(result_image)) 



k = cv2.waitKey(0)
cv2.destroyAllWindows()