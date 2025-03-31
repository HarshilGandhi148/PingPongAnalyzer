import cv2
import time
import numpy as np

video_path = "PingPongVid.mp4"
video = cv2.VideoCapture(video_path)
window_name = "Ball Detection"
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
paused = False

#color thresholds for ball detection
lower_white = np.array([0, 0, 170])
upper_white = np.array([180, 50, 255])
lower_orange = np.array([5, 150, 150])
upper_orange = np.array([25, 255, 255])

def detect_ball(image, prev):
    #grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev = cv2.GaussianBlur(prev, (5, 5), 0)

    #difference
    img = cv2.absdiff(prev, img)

    #remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img,30,255,cv2.THRESH_BINARY)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)))
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    #color mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create color masks
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Combine color and motion masks
    img = cv2.bitwise_and(img, cv2.bitwise_or(mask_white, mask_orange))

    #contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    filtered_cnts = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # 0.7 - 1.2

        if 30 < area < 100 and 30 < perimeter < 150 :
            filtered_cnts.append(cnt)

    new_img = 0
    for c in filtered_cnts:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        new_img = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    return new_img

_, previous = video.read()
while True:
    if not paused:
        result, vid = video.read()
        if result is False: break

    processed = detect_ball(vid, previous)
    previous = vid

    cv2.imshow(window_name, vid)
    k = cv2.waitKey(1) & 0xFF
    #esc
    if k == 27:
        break
    elif k == 32:
        paused = not paused

video.release()
cv2.destroyAllWindows()