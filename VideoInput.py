import numpy as np
import cv2 as cv


# Of main frame
radius =  int(300)
def get_nextframe():
    returned, frame = cap.read()
    if not returned:
        return False, None
    else:
        frame = frame[0:0+radius, 0:0+radius]
        return returned, frame
cap = cv.VideoCapture('ignore/Gameplay.mp4')

returned, trackingframe = get_nextframe()



preview = trackingframe
trackingframe = cv.cvtColor(trackingframe, cv.COLOR_BGR2GRAY)

# Skip some frames
for i in range(1,90):
    get_nextframe()

while cap.isOpened():
    returned, trackingframe = get_nextframe()
    preview = trackingframe
    trackingframe = cv.cvtColor(trackingframe, cv.COLOR_BGR2GRAY)

    # Crom trackingfram into becomming circlecrop


    cv.circle(trackingframe, (radius//2,radius//2), 16, (0,255,0))

    cv.imshow('preview', preview)
    cv.imshow('Tracking', trackingframe)


    cv.waitKey(1)


cap.release()
cv.destroyAllWindows()