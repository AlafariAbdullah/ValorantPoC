###
"""
my attempt to stop rotations
"""
###

import numpy as np
import cv2 as cv

# Of main frame
sides =  int(250)
offset = 20
def get_nextframe():
    returned, frame = cap.read()
    if not returned:
        return False, None
    else:
        frame = frame[offset:offset+sides, offset:offset+sides]
        return returned, frame
cap = cv.VideoCapture('Media/Gameplay.mp4')

returned, trackingframe = get_nextframe()



preview = trackingframe
trackingframe = cv.cvtColor(trackingframe, cv.COLOR_BGR2GRAY)


# Skip some frames
for i in range(1,421):
    get_nextframe()

while cap.isOpened():
    returned, trackingframe = get_nextframe()
    preview = trackingframe
    # Turn black and white
    trackingframe = cv.cvtColor(trackingframe, cv.COLOR_BGR2GRAY)

    # crop trackingfram into becomming circlecrop
    mask = np.zeros_like(trackingframe, dtype=np.uint8)
    cv.circle(mask , (sides//2,sides//2), 130, (255,255,255), -1) # -1 means fill and 130 (x,y) is center, and 130 is radius
    trackingframe = cv.bitwise_and(trackingframe, trackingframe, mask=mask)





    # # ---- 1) Color mask to keep gray minimap only (before grayscale)
    lab = cv.cvtColor(preview, cv.COLOR_BGR2Lab)
    L, A, B = cv.split(lab)
    grayish = (cv.absdiff(A,128) + cv.absdiff(B,128)) < 18   # loosen/tighten 18
    bright  = (L > 60) & (L < 240)
    color_mask = (grayish & bright).astype(np.uint8)         # 0/1
    # restrict color mask to circular ROI as well
    color_mask = cv.bitwise_and(color_mask, color_mask, mask=mask)
    # apply mask to original BGR, then convert THAT to gray for edges
    masked_bgr = cv.bitwise_and(preview, preview, mask=color_mask * 255)
    trackingframe = cv.cvtColor(masked_bgr, cv.COLOR_BGR2GRAY)
    # #     # visualize the LAB components and combined mask
    # cv.imshow('L_channel', L)
    # cv.imshow('A_channel', A)
    # cv.imshow('B_channel', B)
    # cv.imshow('grayish_mask', (grayish.astype(np.uint8) * 255))
    # cv.imshow('bright_mask', (bright.astype(np.uint8) * 255))
    # cv.imshow('combined_color_mask', (color_mask.astype(np.uint8) * 255))








    # Utilize edges to measure
    trackingframe = cv.GaussianBlur(trackingframe, (3,3), 0) # blur To improve accuracy 


    edges = cv.Canny(trackingframe, 5, 100) # adjusted to make detection as best

    # --- combine color mask with edges (restrict edges to gray minimap only) ---
    edges_color = cv.bitwise_and(edges, edges, mask=(color_mask * 255))
        # (optional) seal tiny gaps for cleaner contours
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    edges_color = cv.morphologyEx(edges_color, cv.MORPH_CLOSE, k)
    cv.imshow('edges_color', edges_color)




















    cv.imshow('edges', edges)







    # cv.imshow('Tracking', trackingframe)
    cv.imshow('preview', preview)


    cv.waitKey(1)


cap.release()
cv.destroyAllWindows()