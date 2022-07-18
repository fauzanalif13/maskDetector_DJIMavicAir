from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from face_detection import RetinaFace
from threading import Thread

#mendeteksi dalam video
pathVideo = r'D:\CODING SKRIPSI\Python- DJI Mavic Air\faceMask Detector\video_testing\DJI_0396.MP4'
pathDashcam = 'tcp://193.168.0.1:6200/'
pathDrone = 'rtmp://127.0.0.1:1935'
# w = 1080
# h = 720

# cap = cv2.VideoCapture(path)
# cap = cv2.VideoCapture()
# cap.open('rtmp://127.0.0.1:1935')
detector = RetinaFace()
thresh = 0.56

def detect_and_predict_mask(frame, maskNet):
    # membuat dan menentukan dimensi deteksi wajah
    (h, w) = frame.shape[:2] #dari indeks 0 hingga indeks 1, mengambil 2 element pertama
    #kita cuma mengambil 2 indeks, yaitu indeks height dan width
    detections = detector(frame)

    # untuk menampung faces, locs, preds yg didapatkan
    faces = []
    locs = []
    preds = []

    for face in detections:
        # confidence dalam pendeteksian
        box, landmarks, confidence = face #memasukkan semua nilai box, ldmrks, dan cnfdnc dari variabel face

        # menentukan confidence, jika confidence lebih besar dari args
        if confidence >= thresh:
            # memberikan bounding box
            # box = int(max(0, box[0])), int(max(0, box[1])), int(max(0, box[2])), int(max(0, box[3]))
            (startX, startY, endX, endY) = box.astype("int") #bounding box yg diconvert ke bentuk int

            # memastikan bounding box berada di wajah
            (startX, startY) = (max(0, startX), max(0, startY)) #outputnya bisa jadi mines, jadi diambil nilai 0
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY)) #karena 256-1

            # convert deteksi face
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face) #convert PIL image to numpy array
            face = preprocess_input(face) #memberikan identity number ke setiap batch image, dari 3d ke 4d

            # mengintegrasikan bounding box ke face detector
            faces.append(face)
            locs.append((startX, startY, endX, endY))
        else:
            print("rejected", confidence)
    # membuat prediksi jika ada wajah yang ditemukan, minimal 1
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32") #mengubah nilai warna pixel int ke float
        preds = maskNet.predict(faces, batch_size=32) #jumlah komputasi yang dapat dilakukan dalam 1x proses

    return (locs, preds)


# menload model dari hdd
print("Sedang membuka model..,")
maskNet = load_model("model/mask_detector.model")

print("Memulai video streaming melalui drone...")

class ThreadedCamera(object):
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)

        self.thread = Thread(target=self.update, args=()) #membuat threading
        self.thread.daemon = True #jenis thread
        self.thread.start() #memulai thread

        # fps_input_stream = int(self.capture.get(5))  # hardware fps
        # print("FPS of input stream: {}".format(fps_input_stream))

        self.status = False #aktif atau tdk kamera
        self.frame = None #tdk ada frame

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def grab_frame(self):
        if self.status:
            return self.frame
        return None

if __name__ == '__main__':
    stream_link = pathDrone
    streamer = ThreadedCamera(stream_link)


#save video streaming
cap = cv2.VideoCapture(pathDrone)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)+ 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
# print ('Size video yang ditangkap: '+ width, height)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, 2.4, size)

while True:
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
            # isUseMask = withoutMask < 0.75
            isUseMask = mask > 0.56 & withoutMask < 0.75
            # label = "Bermasker" if mask > withoutMask else "Tidak bermasker"
            # color = (0, 255, 0) if label == "Bermasker" else (0, 0, 255)
            label = "Bermasker" if isUseMask else "Tidak bermasker"
            color = (0, 255, 0) if label == "Bermasker" else (0, 0, 255)


            # menuliskan probability ketika wajah terdeteksi
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # output
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        #save video hasil deteksi
        out.write(frame)

        #fps
        frame = cv2.resize(frame, (width, height))
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Gagal Mendeteksi.")

out.release()
print("Prototyping selesai, VIdeo recording telah disimpan.")
cv2.destroyAllWindows()
