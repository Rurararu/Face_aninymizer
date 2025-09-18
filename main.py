import os.path
import argparse
import cv2
import mediapipe as mp

def process_img(img, face_detection):

    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_date = detection.location_data
            bbox = location_date.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 5)
            # blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (50, 50))

    return img

args = argparse.ArgumentParser()
args.add_argument('--mode', default='image')
args.add_argument('--filePath', default='./train/testimg.jpg')
# args.add_argument('--filePath', default='./data/testvideo.mp4')


args = args.parse_args()

# read image
# img = cv2.imread(img_path)
# H, W = img.shape[:2]

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection( model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode in ('image'):
        img = cv2.imread(args.filePath)

        img = process_img(img, face_detection)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        # save image
        cv2.imwrite(os.path.join('./data/testimg_blur.jpg'), img)

    elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(os.path.join('./data/output.mp4'),
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       25, (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_img(frame, face_detection)

            output_video.write(frame)

            ret, frame = cap.read()

        cap.release()
        output_video.release()
