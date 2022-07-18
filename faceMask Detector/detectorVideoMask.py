from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from face_detection import RetinaFace
import numpy as np
import cv2

global img


detector = RetinaFace()
cap = cv2.VideoCapture(1)
thresh = 0.56


def detect_and_predict_mask(frame, maskNet):
    # membuat dan menentukan dimensi deteksi wajah
    (h, w) = frame.shape[:2]

    detections = detector(frame)

    # menentukan lokasi face
    faces = []
    locs = []
    preds = []

    for face in detections:
        # confidence dalam pendeteksian
        box, landmarks, confidence = face

        # menentukan confidence, jika confidence lebih besar dari args
        if confidence >= thresh:
            # memberikan bounding box
            # box = int(max(0, box[0])), int(max(0, box[1])), int(max(0, box[2])), int(max(0, box[3]))
            (startX, startY, endX, endY) = box.astype("int")

            # memastikan bounding box berada di wajah
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # convert deteksi face
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # mengintegrasikan bounding box ke face detector
            faces.append(face)
            locs.append((startX, startY, endX, endY))
        else:
            print("rejected", confidence)
    # membuat prediksi jika ada wajah yang ditemukan, minimal 1
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


# menload model dari hdd
print("Sedang membuka model..,")
maskNet = load_model("model/mask_detector.model")

print("Memulai video streaming melalui drone...")

while True:
    success, frame = cap.read()

    # menentukan face memakai masker atau tidak
    (locs, preds) = detect_and_predict_mask(frame, maskNet)

    # jika ada pendeteksi face, loop
    for (box, pred) in zip(locs, preds):
        # unpack prediction yang telah dibuat
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # memberikan warna dan label kepada pendeteksian
        label = "Bermasker" if mask > withoutMask else "Tidak bermasker"
        color = (0, 255, 0) if label == "Bermasker" else (0, 0, 255)

        # menuliskan probability ketika wajah terdeteksi
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # output
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # fps
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Prototyping selesai.")
cv2.destroyAllWindows()
