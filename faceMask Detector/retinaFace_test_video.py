from face_detection import RetinaFace
import numpy as np
import cv2
from imutils.video import FPS

color = (0,255,0)
fps = FPS().start()
thresh = 0.97

# video
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture()
# cap.open('rtmp://127.0.0.1:1935')
detector = RetinaFace()

while (cap.isOpened()):
    ret, img = cap.read()

    faces = detector(img)

    if faces is not None:
        for face in faces:
            box, landmarks, score = face
            if score >= thresh:
                box = int(max(0, box[0])), int(max(0, box[1])), int(max(0, box[2])), int(max(0, box[3]))
                color = (0, 0, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    landmark5 = landmarks.astype(np.int)
                    # print(landmark.shape)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

        cv2.imshow('Frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
