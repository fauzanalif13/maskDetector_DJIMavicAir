from retinaface import RetinaFace
import numpy as np
import cv2
w = 1008 #hasil bagi 4 size dari img test.jpg
h = 756
color = (0,255,0)

#image path
path = r'G:\CODING SKRIPSI\Python- DJI Mavic Air\faceMask Detector\image_testing\test.jpg'
image = cv2.imread(path)

obj = RetinaFace.detect_faces(path)

# len(obj.keys())
#utk print brp parameter wajah

while True:
    for key in obj.keys():
        identity = obj[key]
        # print identity

        label = 'Face'

        facial_area = identity['facial_area']

        cv2.putText(image, label, (facial_area[0], facial_area[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
        cv2.rectangle(image, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), color, 2)

    image_resize = cv2.resize(image,(w,h))
    cv2.imshow("Frame", image_resize)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break