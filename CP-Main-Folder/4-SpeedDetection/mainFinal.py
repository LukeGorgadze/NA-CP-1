import time
import cv2
import math

cap = cv2.VideoCapture('Vids\Ramazvid.mp4')

prev_gray = None
prev_pos = None
distance = None
fps = cap.get(cv2.CAP_PROP_FPS)

startTime = time.time()
entered = False
leftBox = (0,0)
escaped = False
rightBox = (0,0)

movesFromLeft = False
movesFromRight = False


frame_diff = None
prev_edges = None
while True:
    cv2.waitKey(20)
    ret, frame = cap.read()
    if not ret:
        break

    xx = 400
    yy = 150

    xx2 = 1250
    yy2 = 100

    # Step 1 apply gaussian to bw image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    cv2.rectangle(frame, (xx,yy), (xx+40, yy+40), (255, 0, 0), 1)
    cv2.rectangle(frame, (xx2,yy2), (xx2+40, yy2+40), (0, 0, 255), 1)

    # Step 2 apply laplacian to blurred image
    # edges = cv2.Laplacian(gray, cv2.CV_8UC1)
    # apply sobel to blurred image
    edges = cv2.Sobel(gray, cv2.CV_8UC1, 1, 1, ksize=5)
    edges = cv2.threshold(edges, 60, 255, cv2.THRESH_BINARY)[1]

    # Step 3 calculate difference of edge frames
    if prev_edges is not None:
        frame_diff = cv2.absdiff(edges, prev_edges)
        # frame_diff = cv2.GaussianBlur(frame_diff, (11, 11), 0)
        # Step 4 Find the contours in the edge image
        contours, _ = cv2.findContours(frame_diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Step 5 Iterate through the contours and find the car
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 1)
            
    prev_edges = edges.copy()

    if frame_diff is not None:
        cv2.imshow('frame2', frame_diff)
        cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

moveTime = 0
if movesFromLeft:
    moveTime = rightBox[1] - leftBox[1]
elif movesFromRight:
    moveTime = leftBox[1] - rightBox[1]

if moveTime <= 0:
    moveTime = 1
print(moveTime,"Segment Time interval")
print(30 / moveTime, "M / S")
print(30 / moveTime * 3.6, "KM / H")
# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()