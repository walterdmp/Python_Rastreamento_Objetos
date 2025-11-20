import cv2
import sys

def create_tracker_by_name(tracker_type):
    if tracker_type == 'BOOSTING':
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        return cv2.legacy.TrackerMIL_create()
    elif tracker_type == 'KCF':
        return cv2.legacy.TrackerKCF_create()
    elif tracker_type == 'TLD':
        return cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        return cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == 'MOSSE':
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        return cv2.legacy.TrackerCSRT_create()
    else:
        return cv2.legacy.TrackerCSRT_create()

print("Selecione o Tracker:")
print("1 - CSRT")
print("2 - KCF")
print("3 - MOSSE")
print("4 - MedianFlow")
print("5 - MIL")
print("6 - Boosting")
print("7 - TLD")

choice = input("Digite o numero: ")

trackers_map = {
    '1': 'CSRT', '2': 'KCF', '3': 'MOSSE', 
    '4': 'MEDIANFLOW', '5': 'MIL', '6': 'BOOSTING', '7': 'TLD'
}

if choice in trackers_map:
    selected_tracker = trackers_map[choice]
else:
    selected_tracker = 'CSRT'

video_path = "nascar.mp4"
cap = cv2.VideoCapture(video_path)

multi_tracker = cv2.legacy.MultiTracker_create()

if not cap.isOpened():
    sys.exit()

success, frame = cap.read()
if not success:
    sys.exit()

frame = cv2.resize(frame, (1020, 720))

bbox = cv2.selectROI('Tracking', frame, fromCenter=False)
tracker = create_tracker_by_name(selected_tracker)
multi_tracker.add(tracker, frame, bbox)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (1020, 720))

    success, boxes = multi_tracker.update(frame)

    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        cv2.putText(frame, f"{selected_tracker} ID:{i}", (p1[0], p1[1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Tracking', frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('s'):
        roi = cv2.selectROI('Tracking', frame, fromCenter=False)
        if roi != (0, 0, 0, 0):
            tracker = create_tracker_by_name(selected_tracker)
            multi_tracker.add(tracker, frame, roi)

    elif key == ord('f'):
        cv2.imwrite("frame_capturado.jpg", frame)

    elif key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()