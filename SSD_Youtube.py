import numpy as np
import cv2
import pafy
import time


# Params
PROTOTXT = "MobileNetSSD_deploy.prototxt" # Specify path for .prototxt and .caffemodel
MODEL = "MobileNetSSD_deploy.caffemodel"

URL =  "https://youtu.be/APvT4qVKfRQ" # Youtube url

CONF_THRES = 0.4 # Confidence threshold for making prediction


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",  "car", 
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

ENABLE_GPU = 0

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))



def load_model():

    # Read model
    model = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    if ENABLE_GPU:
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    print("[INFO] Loaded model")

    return model


def subscribe_stream():

    v = pafy.new(URL)
    
    stream = v.getbest(preftype="any")

    
    cap = cv2.VideoCapture(stream.url)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    print ("[INFO] Subscribed to stream. Resolution: ",stream.resolution)

    return cap



if __name__ == "__main__":
    
    model = load_model()

    video = subscribe_stream()

    cv2.namedWindow('output', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('output',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
   
    while video.isOpened():

        ret, frame = video.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        model.setInput(blob)
        detections = model.forward()
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > CONF_THRES:
                idx = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(CLASSES[idx],confidence*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        cv2.imshow('output',frame)
      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
