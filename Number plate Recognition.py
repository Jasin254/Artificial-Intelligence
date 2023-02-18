import cv2
import imutils
import pytesseract

# load the image
image = cv2.imread("car.jpg")

# resize the image and convert it to grayscale
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply thresholding to the image
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)
thresh = cv2.adaptiveThreshold(edged, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# find contours in the image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# loop over the contours to find the number plate
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    
    # check if the contour has four vertices
    if len(approx) == 4:
        screenCnt = approx
        break

# if we have found the number plate
if screenCnt is not None:
    # apply a perspective transform to the image to get a top-down view of the number plate
    warped = imutils.four_point_transform(gray, screenCnt.reshape(4, 2))
    output = pytesseract.image_to_string(warped, config='--psm 11')
    
    # print the number plate
    print("Number Plate Detected: ", output)
    
    # draw a rectangle around the number plate
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)

# display the image
cv2.imshow("Number Plate Recognition", image)
cv2.waitKey(0)
