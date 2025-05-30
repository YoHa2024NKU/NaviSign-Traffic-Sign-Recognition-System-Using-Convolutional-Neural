import cv2

cap = cv2.VideoCapture(0)  # Open the default camera
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame.")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
cap.release()
cv2.destroyAllWindows()