import cv2

# Read the video
video_path = 'camera_1.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize the tracker
tracker = cv2.TrackerMedianFlow_create()

# Read the first frame
ret, frame = cap.read()
bbox = cv2.selectROI("Select Object", frame, False)
tracker.init(frame, bbox)

# Loop through frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update the tracker
    success, bbox = tracker.update(frame)
    
    # Draw bounding box
    if success:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Frame', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
