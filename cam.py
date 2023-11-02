import cv2

cam = cv2.VideoCapture(1)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("cam", frame)

    k = cv2.waitKey(1)
    if k == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k == 32:
        # SPACE pressed
        cv2.imwrite("usb_cam.png", frame)

cam.release()
cv2.destroyAllWindows()
