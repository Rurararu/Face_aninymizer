import os.path
import cv2
import mediapipe as mp

def process_web(frame, face_detection):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = face_detection.process(frame_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_date = detection.location_data
            bbox = location_date.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if x2 > x1 and y2 > y1:
                face_region = frame[y1:y2, x1:x2]
                face_region = cv2.blur(face_region, (25, 25))
                frame[y1:y2, x1:x2] = face_region

            # frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 5)
            # blur faces
            # frame[y1:y1 + h, x1:x1 + w, :] = cv2.blur(frame[y1:y1 + h, x1:x1 + w, :], (25, 25))

    return frame

cap = cv2.VideoCapture(0)
# H, W = cap.shape[:2]
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

mp_face_detection = mp.solutions.face_detection



with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()

        frame = process_web(frame, face_detection)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
