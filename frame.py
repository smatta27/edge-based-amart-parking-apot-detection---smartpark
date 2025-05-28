import cv2

cap = cv2.VideoCapture('carPark.mp4')
ret, frame = cap.read()  # Reads the first frame
if ret:
    cv2.imwrite('video_frame.jpg', frame)
cap.release()