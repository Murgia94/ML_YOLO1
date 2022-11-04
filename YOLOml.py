import numpy as np
import argparse
import cv2
from time import time

def parse_args():
    ap = argparse.ArgumentParser(description='This is a demo that perform object recognition throught your own webcam.'
                                             'The pretrained model is traind on COCO dataset.')
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.5,
                    help="NMS threshold ")
    ap.add_argument("--weights", type=str, default= "yolo_coco/yolov3.weights",
                    help="Path network configuration ")
    ap.add_argument( "--config", type=str, default="yolo_coco/yolov3.cfg",
                    help="configuration file of the network (file.cfg)")
    ap.add_argument( "--height", type=int, default=128,
                    help="height input image")
    ap.add_argument( "--width", type=int, default=128,
                    help="width input image")
    args = ap.parse_args()

    return args


def main():
    args = parse_args()
    weightsPath = args.weights
    configPath = args.config
    # load the COCO labels
    labelsPath = "yolo_coco/coco.names"
    labels = open(labelsPath).read().strip().split("\n")
    # initialize a random list of colors to represent each possible class label
    # in our case are the 80 classes from the COCO dataset
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(labels), 3),
                               dtype="uint8")
    input_img_shape = [args.height, args.width]
    # load our YOLO object detector
    print("[INFO] loading YOLO")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    layer_names = net.getLayerNames()
    layer_names = net.getUnconnectedOutLayersNames()
    # initialize the video stream
    vs = cv2.VideoCapture(0)
    (W, H) = (None, None)
    # loop over frames from the webcam
    while True:
        # read the next frame from the webcam
        ret, frame = vs.read()
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # inizialize the time to mesure inference time
        tic = time()
        # construct a blob from the input frame (preprocess the image, normalizeed to 1 without cropping)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, input_img_shape,
                                     swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(layer_names)

        toc = time()
        elapsed_time = toc - tic
        print("[INFO] FPS: {}".format(1/elapsed_time))
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
               # extract the class ID and confidence
                scores = detection[5:]  # from 5 because the first four values are to build the bbox and the 4th
                                        # to be soure that an object is present.
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > args.confidence:
                    # scale the bounding box coordinates back relative to
                    # the size of the original image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # apply non-maxima suppression to suppress weak or overlapped
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence,
                                args.threshold)
        # check if at least one detection exists
        if len(idxs) > 0:
            # loop over the detections
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(labels[classIDs[i]],
                                           confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame with labels and bounding boxes
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the camera
    print("[INFO] cleaning")
    vs.release()


if __name__ == '__main__':
    main()
