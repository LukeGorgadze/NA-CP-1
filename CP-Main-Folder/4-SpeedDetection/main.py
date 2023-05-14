import time
import cv2
import math

# Open the video file
cap = cv2.VideoCapture('Vids\Ramazvid.mp4')

# Initialize variables
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
while True:
    cv2.waitKey(20)
    # Read the next frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    xx = 400
    yy = 150

    # Set 2 comparation points
    xx2 = 1250
    yy2 = 100

    # Convert the frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute the difference between the current and previous frame
    frame_diff = None

    cv2.rectangle(frame, (xx,yy), (xx+40, yy+40), (255, 0, 0), 1)
    cv2.rectangle(frame, (xx2,yy2), (xx2+40, yy2+40), (0, 0, 255), 1)
    if prev_gray is not None:
        frame_diff = cv2.absdiff(gray, prev_gray)

        # Apply thresholding to the frame difference
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

        # Apply Canny edge detection to the thresholded image
        # edges = cv2.Canny(thresh, 50, 150)

        # Apply soble edge detection
        edges = cv2.Laplacian(thresh, cv2.CV_8UC1)

        # Find the contours in the edge image
        contours, _ = cv2.findContours(
            edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through the contours and find the car
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # Check if the contour is likely a car
            if aspect_ratio > 2 and aspect_ratio < 4:
                # Calculate the center position of the car
                curr_pos = (x + w // 2, y + h // 2)

                if not movesFromLeft and not entered and curr_pos[0] < xx and curr_pos[1] < yy:
                    movesFromLeft = True
                
                if not movesFromRight and not entered and curr_pos[0] > xx and curr_pos[1] > yy:
                    movesFromRight = True

                if movesFromLeft:
                    if not entered and curr_pos[0] > xx and curr_pos[1] > yy:
                        print("CAR IS MOVING FROM LEFT TO RIGHT")
                        entered = True
                        leftBox = (curr_pos,time.time() - startTime)

                    if not escaped and curr_pos[0] > xx2 and curr_pos[1] > yy2:
                        escaped = True
                        rightBox = (curr_pos,time.time() - startTime)
                
                if movesFromRight:
                    if not entered and curr_pos[0] < xx and curr_pos[1] < yy:
                        print("CAR IS MOVING FROM RIGHT TO LEFTs")
                        entered = True
                        leftBox = (curr_pos,time.time() - startTime)

                    if not escaped and curr_pos[0] < xx2 and curr_pos[1] < yy2:
                        escaped = True
                        rightBox = (curr_pos,time.time() - startTime)


    # Display the resulting frame
    if frame_diff is not None:
        cv2.imshow('frame', gray)
        cv2.imshow('frame2', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

    # Update the previous frame and position
    prev_gray = gray.copy()

moveTime = 0
if movesFromLeft:
    moveTime = rightBox[1] - leftBox[1]
elif movesFromRight:
    moveTime = leftBox[1] - rightBox[1]
print(moveTime,"Segment Time interval")
print(30 / moveTime, "M / S")
print(30 / moveTime * 3.6, "KM / H")
# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()