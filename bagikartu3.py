import cv2
import numpy as np
from stackImages import stackImages
import os

def empty(a):
    pass

cards = {}
cards['user'] = []
cards['computer'] = []
turn = 'user'
path = 'dataset'
templates = {}
for filename in os.listdir(path):
    card_name = os.path.splitext(filename)[0]
    card_image = cv2.imread(os.path.join(path, filename), cv2.COLOR_BGR2GRAY)
    card_image = card_image[0:200, 0:100]
    templates[card_name] = card_image

widthImg = 480
heightImg = 640

def greenScreen(cam):
    imgHSV = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    mask = cv2.bitwise_not(mask)
    greenscreen = cv2.bitwise_and(cam, cam, mask=mask)

    return greenscreen, mask

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    imgDilate = cv2.dilate(imgCanny, np.ones((5, 5)), iterations=3)
    imgErode = cv2.erode(imgDilate, np.ones((5, 5)), iterations=2)
    return imgErode

def findEdgePoints(img):
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        # print("area: ",area)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            # print(approx.shape)
            if len(approx) == 4:
                # print("approx: ", approx)
                # cv2.drawContours(imgCnt, approx, -1, (255, 0, 0), 20)
                # print("points: ", point)
                return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print("add: ", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew

def getWarp(img, points):
    if not points is None:
        points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32(
        [
            [0, 0],
            [widthImg, 0],
            [0, heightImg],
            [widthImg, heightImg]
        ]
    )
    # print(pts1)
    if points is None or not points.any():
        return np.zeros((widthImg, heightImg))
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warpp = cv2.warpPerspective(imgCnt, matrix, (widthImg, heightImg))
        cv2.imshow("warp", preProcessing(warpp[0:200, 00:100]))
        if not warpp.any():
            cv2.destroyWindow("warp")
        return warpp

def getSAD(warpcard):
    if warpcard.size == 0:
        return "No card detected"
    min = np.Infinity
    card_match = ""
    warpcard = cv2.cvtColor(warpcard, cv2.COLOR_BGR2GRAY)
    warpcard = warpcard[0:200, 0:100]
    for card_name, card_image in templates.items():
        databasecard = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(databasecard, (warpcard.shape[1], warpcard.shape[0]))
        sad = np.sum(cv2.absdiff(warpcard, resize))
        if sad < min:
            min = sad
            card_match = card_name
    return card_match


cap = cv2.VideoCapture(1)
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 49, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
while True:
    _, img = cap.read()
    imgCnt = img.copy()
    blank = np.zeros_like(img)
    green_screen, mask = greenScreen(img)
    preprocessed = preProcessing(green_screen)
    getPoint = findEdgePoints(preprocessed)
    warp = getWarp(img, getPoint)
    if getPoint is not None:
        card_found = getSAD(warp)
        cv2.putText(imgCnt, card_found, reorder(getPoint).reshape(4, 2).tolist()[0], cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 0, 0), 2)
        # if turn == 'user':
        #     if len(cards[turn]) < 4:
        #         if card_found in cards[turn]:
        #             continue
        #         cards[turn].append(card_found)
        #     else:
        #         turn = 'computer'
        # else:
        #     if len(cards[turn]) < 4:
        #         if card_found in cards[turn]:
        #             continue
        #         cards[turn].append(card_found)
        #     else:
        #         break
        # print(card_found)
    cv2.imshow("Result", stackImages(0.3, ([img, green_screen, imgCnt], [mask, preprocessed, blank])))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


