import cv2
from cv2 import threshold
import pafy

import numpy as np
import time
import cv2
import os




# YOLO 
if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--input", required=True,
    #     help="path to input video")
    # ap.add_argument("-c", "--confidence", type=float, default=0.5,
    #     help="minimum probability to filter weak detections")
    # ap.add_argument("-t", "--threshold", type=float, default=0.3,
    #     help="threshold when applying non-maxima suppression")
    # args = vars(ap.parse_args())
    

    # Parameters 
    confidence_thres = 0.5
    non_max_thres    = 0.3


    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join(["/home/michael/YoutubeRealTimeYOLO/darknet/data/coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join(["/home/michael/YoutubeRealTimeYOLO/darknet/weights/yolov3.weights"])
    configPath  = os.path.sep.join(["/home/michael/YoutubeRealTimeYOLO/darknet/cfg/yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    writer = None
    (W, H) = (None, None)


    # ------------------------------------ 
    url = "https://www.youtube.com/watch?v=bhWdPoWJzCE"
    v = pafy.new(url)

    best = v.getbest(preftype="any")


    cap = cv2.VideoCapture(best.url)
    if (cap.isOpened()==False):
        print("Could not open the video")


    # loop over frames from the video file stream
    while cap.isOpened():
        # read the next frame from the file
        ret, frame = cap.read()
        
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not ret:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > confidence_thres:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thres,non_max_thres)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('outputWindows',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):# Press 'ESC' for exiting video
            break 


    cap.release()

    cv2.destroyAllWindows()

    # --------------------------------------------------------------------------------------------------------------
    # url = "https://www.youtube.com/watch?v=bhWdPoWJzCE"
    # v = pafy.new(url)

    # best = v.getbest(preftype="any")



    # # print(v.title)
    # # print(v.duration)
    # # print(v.author)
    # # print(v.length)
    # # print(v.thumb)
    # # print(v.videoid)
    # # print(v.viewcount)



    # cap = cv2.VideoCapture(best.url)
    # if (cap.isOpened()==False):
    #     print("Could not open the video")

    # while (cap.isOpened()):

    #     ret, frame = cap.read()

    #     if ret == True:
    #         cv2.imshow("Frame",frame)

        
    #         if cv2.waitKey(50) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break

    # # ADD YOLO STUFF HERE


    # cap.release()

    # cv2.destroyAllWindows()

