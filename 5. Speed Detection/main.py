import cv2
import math

# Open the video file
cap = cv2.VideoCapture('Vids\Ramazvid.mp4')

# Initialize variables
prev_gray = None
prev_pos = None
distance = None
fps = cap.get(cv2.CAP_PROP_FPS)
theta = math.radians(-60)  # Camera angle around the z-axis (in radians)
car_height = 1.5  # Height of the car in meters

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute the difference between the current and previous frame
    if prev_gray is not None:
        frame_diff = cv2.absdiff(gray, prev_gray)

        # Apply thresholding to the frame difference
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]

        # Apply Canny edge detection to the thresholded image
        edges = cv2.Canny(thresh, 50, 150)

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

                # Calculate the distance between the car and the camera
                if distance is None:
                    distance = (
                        car_height * frame.shape[0]) / (2 * h * math.tan(theta))

                # Calculate the speed of the car
                if prev_pos is not None:
                    speed = distance * fps * \
                        math.tan(theta) / \
                        (2 * max(abs(curr_pos[1] - prev_pos[1]), 1))

                    # Draw the speed on the frame
                    cv2.putText(frame, "Speed: {:.2f} km/h".format(speed * 3.6),
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print(speed)

                # Update the previous position
                prev_pos = curr_pos

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # Update the previous frame and position
    prev_gray = gray.copy()

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()