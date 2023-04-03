import cv2 as cv
import numpy as np
import os

def readImages(img_dir):
    file_list = os.listdir(img_dir)

    image_array = []

    for img in file_list:
        # Construct the full path to the image file
        image_path = os.path.join(img_dir, img)
        image = cv.imread(image_path)
        
        # Get tray info
        trayX, trayY = defineTray(image)
        # Get circles info
        circles = defineCircles(image)

        # Calculate coins
        totalInside, totalOutside, coinsInside, coinsOutside = calculateCoins(trayX, trayY, circles)
        total = totalInside + totalOutside
        cv.putText(image, f'Total inside: {totalInside}', (500,800), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(image, f'Total outside: {totalOutside}', (500,840), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(image, f'Total: {total}', (500,880), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(image, f'Coins inside: {coinsInside}', (500,920), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(image, f'Coins outside: {coinsOutside}', (500,960), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

        #Draw contours
        drawTrayContours(image, trayX, trayY)
        drawCirclesContours(image, circles)
        
        resized_image = cv.resize(image, (0, 0), fx=0.4, fy=0.4)
        image_array.append(resized_image)
    
    
    first_half = image_array[:len(image_array)//2]
    second_half = image_array[len(image_array)//2:]

    top_row = np.hstack(first_half)
    bottom_row = np.hstack(second_half)

    image_stack = np.vstack((top_row, bottom_row))

    cv.imshow('All Images', image_stack)
    cv.waitKey(0)
    cv.destroyAllWindows()

def calculateCoins(trayX, trayY, circles):
    totalInside = 0
    totalOutside = 0
    coinsInside = 0
    coinsOutside = 0
    #Values to calculate totals
    minCoinsInside = 0
    maxCoinsInside = 0
    minCoinsOutside = 0
    maxCoinsOutside = 0

    #Tray coordinates
    hX = max(trayX)
    hY = max(trayY)
    lX = min(trayX)
    lY = min(trayY)

    #Radius of max array on each image
    max_radius = 0
    for circle in circles[0,:]:
        _, _, radius = circle
        if radius > max_radius:
            max_radius = radius

    # center = (i[0], i[1])
    # radius = i[2] 
    for i in circles[0,:]:
        centerX = i[0]
        centerY = i[1]
        radius = i[2]

        #Count inside
        if centerX > lX and centerX < hX and centerY > lY and centerY < hY:
            coinsInside += 1
            if max_radius - 3.5 < radius:
                maxCoinsInside += 1
            else:
                minCoinsInside += 1
        #Count outside
        else:
            coinsOutside += 1
            if max_radius - 3.5 < radius:
                maxCoinsOutside += 1
            else:
                minCoinsOutside += 1

    #Calculate PLN value
    totalInside = round(5 * maxCoinsInside + 0.05 * minCoinsInside, 2)
    totalOutside = round(5 * maxCoinsOutside + 0.05 * minCoinsOutside, 2)
        
    return (totalInside, totalOutside, coinsInside, coinsOutside)

def defineCircles(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(gray,5)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1, 10, param1=100, param2=35, minRadius=20, maxRadius=40)
    circles = np.uint16(np.around(circles))

    return circles

def drawCirclesContours(img, circles):
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
        
def defineTray(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=80,maxLineGap=50)

    trayX = []
    trayY = []

    #Get contours(lines) of tray
    for line in lines:
        x1,y1,x2,y2 = line[0]
        trayX.extend([x1, x2])
        trayY.extend([y1, y2])

    return (trayX, trayY)

def drawTrayContours(img, trayX, trayY):
    trayTopLeft = (min(trayX), min(trayY))
    trayDownRight = (max(trayX), max(trayY))
    cv.rectangle(img, trayTopLeft, trayDownRight,(0,255,0),2)

#RESULT
readImages('data')