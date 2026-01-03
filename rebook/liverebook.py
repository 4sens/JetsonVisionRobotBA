import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\noovelUser\Documents\YOLO\rebook\runs\reebok_shoe_v52\weights\best.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame, conf=0.3, iou=0.5, verbose=False)[0]  # conf höher, um FP zu drücken
    WINDOW_NAME = "Reebok Detection"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)   # <-- wichtig!
    cv2.resizeWindow(WINDOW_NAME, 1024, 720)    
    cv2.imshow(WINDOW_NAME, r.plot())

    if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
        break

cap.release()

cv2.destroyAllWindows()