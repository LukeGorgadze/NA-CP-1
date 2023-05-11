import cv2
import numpy as np

# Define the video file path
video_path = "Vids\Ramaza.mp4"

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(video_path)

# Define the threshold for edge detection
threshold1 = 30
threshold2 = 100

# Define the kernel size for morphological operations
kernel_size = (5, 5)

# Define the font for displaying speed on the video
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize variables for tracking car speed
car_speed = 0
prev_car_pos = 0
prev_frame_time = 0

# Loop through each frame of the video
while cap.isOpened():
    # Read the current frame from the video
    ret, frame = cap.read()
    
    # If the frame cannot be read, break out of the loop
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection using the Canny function
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # Apply morphological operations to close small gaps and remove noise
    kernel = np.ones(kernel_size, np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through each contour and check if it's a car
    for cnt in contours:
        # Compute the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Check if the bounding box meets certain criteria to be considered a car
        if w > 50 and h > 50 and w < 300 and h < 300 and w > 1.5*h:
            # Compute the center of the bounding box
            cx = x + w/2
            cy = y + h/2
            
            # Check if the car is moving from left to right
            if cx > prev_car_pos:
                # Compute the speed of the car using the time difference between frames
                curr_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                time_diff = curr_frame_time - prev_frame_time
                car_speed = (cx - prev_car_pos) / time_diff * 0.681818
                print(car_speed)
                # Store the current position and time for the next frame
                prev_car_pos = cx
                prev_frame_time = curr_frame_time
            
            # Draw the bounding box and the speed on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Speed: {:.0f} mph".format(car_speed), (x, y - 10), font, 0.5, (0, 255, 0), 1)
    
    # Display the current frame with bounding boxes and speed
    cv2.imshow("Frame", frame)
    
    # Wait for a key press and exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
