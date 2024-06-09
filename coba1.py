import cv2
import numpy as np
from stackImages import stackImages

cap = cv2.VideoCapture(1)

def empty(a):
    pass

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

cap = cv2.VideoCapture(1)
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 49, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)
while cap.isOpened():
    ret, frame = cap.read()
    greenscreen, mask = greenScreen(frame)
    edgepoint = findEdgePoints(frame)
    print(edgepoint)
    cv2.imshow("Result", stackImages(0.5, ([greenscreen, mask])))
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()