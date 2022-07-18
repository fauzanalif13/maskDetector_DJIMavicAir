from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import sys
from face_detection import RetinaFace
from threading import Thread
import time
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue

# mendeteksi dalam video
pathVideo = r'D:\CODING SKRIPSI\Python- DJI Mavic Air\faceMask Detector\video_testing\DJI_0396.MP4'
pathDashcam = 'tcp://193.168.0.1:6200/'
pathDrone = 'rtmp://127.0.0.1:1935'

# cap = cv2.VideoCapture(path)
# cap = cv2.VideoCapture()
# cap.open('rtmp://127.0.0.1:1935')
detector = RetinaFace()
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
            print("rejected: ", confidence)
    # membuat prediksi jika ada wajah yang ditemukan, minimal 1
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32) #jumlah komputasi yang dapat dilakukan dalam 1x proses

    return (locs, preds)


# menload model dari hdd
print("Sedang membuka model..,")
maskNet = load_model("model/mask_detector.model")

print("Memulai video streaming melalui drone...")


class ThreadedCamera(object):
    def __init__(self, source=0, queue_size=128):

        self.capture = cv2.VideoCapture(source)
        self.stopped = False
        self.Q = Queue(maxsize=queue_size)

        # self.thread = Thread(target=self.update, args=())  # membuat threading
        # self.thread.daemon = True  # jenis thread
        # self.thread.start()  # memulai thread

        fps_input_stream = int(self.capture.get(5))  # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))

        self.status = False  # aktif atau tdk kamera
        self.frame = None  # tdk ada frame

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            # if self.capture.isOpened():
            if not self.Q.full():
                # (self.status, self.frame) = self.capture.read()
                (status, frame) = self.capture.read()
                if not status:
                    self.stop()
                    return
                self.Q.put(frame)

    def grab_frame(self):
        if self.status:
            return self.Q.get()
        return None

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


if __name__ == '__main__':
    stream_link = pathVideo
    streamer = ThreadedCamera(stream_link).start()
    time.sleep(1.0)

while streamer.more():
    frame = streamer.grab_frame()
    if frame is not None:
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
        frame = cv2.resize(frame, (1080, 720))
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Gagal Mendeteksi.")

print("Prototyping selesai.")
cv2.destroyAllWindows()
streamer.stop()
