import cv2 as cv
import numpy as np

def nothing(x):
    pass


cv.namedWindow('Color Detectors')
cv.createTrackbar('LH','Color Detectors',0,179,nothing)
cv.createTrackbar('LS','Color Detectors',0,255,nothing)
cv.createTrackbar('LV','Color Detectors',0,255,nothing)
cv.createTrackbar('UH','Color Detectors',179,179,nothing)
cv.createTrackbar('US','Color Detectors',255,255,nothing)
cv.createTrackbar('UV','Color Detectors',255,255,nothing)

# Colors

white = (255,255,255)
black = (0,0,0)
gray = (122,122,122)

colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorIndex = 0

bpoints = []
gpoints = []
rpoints = []
ypoints = []

paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv.rectangle(paintWindow, (40,1), (140,65), (0, 0, 0), 2)
paintWindow = cv.rectangle(paintWindow, (160,1), (255,65), colors[0], -1)
paintWindow = cv.rectangle(paintWindow, (275,1), (370,65), colors[1], -1)
paintWindow = cv.rectangle(paintWindow, (390,1), (485,65), colors[2], -1)
paintWindow = cv.rectangle(paintWindow, (505,1), (600,65), colors[3], -1)

cv.putText(paintWindow, "CLEAR", (49, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, black, 2, cv.LINE_AA)
cv.putText(paintWindow, "BLUE", (185, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, white, 2, cv.LINE_AA)
cv.putText(paintWindow, "GREEN", (298, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, white, 2, cv.LINE_AA)
cv.putText(paintWindow, "RED", (420, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, white, 2, cv.LINE_AA)
cv.putText(paintWindow, "YELLOW", (520, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, white, 2, cv.LINE_AA)





cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret == True:
        frame = cv.flip(frame, 1)
        hsv =  cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        l_h = cv.getTrackbarPos('LH','Color Detectors')
        l_s = cv.getTrackbarPos('LS', 'Color Detectors')
        l_v = cv.getTrackbarPos('LV', 'Color Detectors')
        u_h = cv.getTrackbarPos('UH', 'Color Detectors')
        u_s = cv.getTrackbarPos('US', 'Color Detectors')
        u_v = cv.getTrackbarPos('UV', 'Color Detectors')

        lower_hsv = np.array([l_h,l_s,l_v])
        upper_hsv = np.array([u_h, u_s, u_v])

        frame = cv.rectangle(frame, (40,1),(140,65),gray, -1)
        frame = cv.rectangle(frame, (160, 1), (255, 65), colors[0],-1)
        frame = cv.rectangle(frame, (275, 1), (370, 65), colors[1], -1)
        frame = cv.rectangle(frame, (390, 1), (485, 65), colors[2], -1)
        frame = cv.rectangle(frame, (505, 1), (600, 65), colors[3], -1)
        cv.putText(frame, 'Clear all', (49,33),cv.FONT_HERSHEY_SIMPLEX,0.5,white,2)
        cv.putText(frame, "BLUE", (185, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, "GREEN", (298, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, "RED", (420, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, "YELLOW", (520, 33), cv.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv.LINE_AA)

        kernel = np.ones((5,5),np.uint8)
        Mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        # Noise Removal
        Mask = cv.erode(Mask, kernel,iterations=1)
        Mask = cv.morphologyEx(Mask,cv.MORPH_OPEN,kernel)
        Mask = cv.dilate(Mask, kernel, iterations=1 )

        cnts,_ = cv.findContours(Mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        center = None

        if len(cnts)> 0:

            cnt = sorted(cnts, key=cv.contourArea, reverse=True)[0]
            ((x,y),raduis) = cv.minEnclosingCircle(cnt)

            cv.circle(frame, (int(x),int(y)),int(raduis),colors[3],2)

            M = cv.moments(cnt)

            center = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

            if center[1] <= 65:
                if 40 <= center[0] <= 140:
                    bpoints = []
                    gpoints = []
                    rpoints = []
                    ypoints = []

                    paintWindow[67:, :, :] = 255

                elif 160 <= center[0] <= 255:
                    colorIndex = 0
                elif 275 <= center[0] <= 370:
                    colorIndex = 1  # Green
                elif 390 <= center[0] <= 485:
                    colorIndex = 2  # Red
                elif 505 <= center[0] <= 600:
                    colorIndex = 3  # Yellow
            else:

                if colorIndex == 0:
                    bpoints.insert(0,center)
                elif colorIndex == 1:
                    gpoints.insert(0, center)
                elif colorIndex == 2:
                    rpoints.insert(0, center)
                elif colorIndex == 3:
                    ypoints.insert(0, center)


            points = [bpoints,gpoints,rpoints,ypoints]
            for i in range(len(points)):
                for j in range(1, len(points[i])):

                    cv.line(frame, points[i][j-1], points[i][j], colors[i],2)
                    cv.line(paintWindow, points[i][j - 1], points[i][j], colors[i], 2)










        cv.imshow('Frame',frame)
        cv.imshow('Paint', paintWindow)
        cv.imshow('Mask',Mask)
        if cv.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()