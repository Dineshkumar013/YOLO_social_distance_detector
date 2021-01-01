#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import cv2
import numpy as np
from scipy.spatial import distance as dist
import imutils
import argparse


# In[3]:




# In[5]:


def detectPeople(frame, net, ln):
    # extract the dimensions of the frame
    (h, w) = frame.shape[:2]
    results = []
    # create a blob frome the image
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 255.0,
                                 size=(416,416),
                                 swapRB=True,
                                 crop=False)
    # forward pass for YOLO object detector, will give a boundig box
    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    # loop to each layer outputs
    for out in outputs:
        for detection in out:
            # extract class ID and confidences from detection
            scores = detection[5:]
            class_ID = np.argmax(scores)
            confidence = scores[class_ID]

            # filter out the people from detection
            # here in YOLO labels person index is 0
            if class_ID == 0 and confidence > 0.3:
                # YOLO detector returns a center(x, y) of a bounding box
                # followed with width and height
                box = detection[0:4] * np.array([w, h, w, h])
                (cX, cY, width, height) = box.astype('int')

                # calaculate top left (x, y) from centerX and Center Y
                x = int(cX - (width / 2))
                y = int(cY - (height / 2))

                # update a list
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                centroids.append((cX, cY))
    # apply NMS- non-max suppression for weak bounding boxes
    idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

    # for one box
    if len(idx) > 0:
        for i in idx.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update the result and return
            res = (confidences[i], (x, y, x+w, y+h), centroids[i])
            results.append(res)

    return results
argparser = argparse.ArgumentParser()

argparser.add_argument('--input', '-i', type=str, default="", help="Path of a input file" )

argparser.add_argument('--output', '-o', type=str, default="", help="Path of a output file" )

argparser.add_argument('--display', '-d', type=int, default=0, help="output of the frame will diplayed" )

args = vars(argparser.parse_args(["--input","C:\\Users\\DINESH HEMANTH\\Downloads\\SOCIAL_DISTANCE_DETECTOR\\pedestrians.mp4","--output","my_output.avi","--display","1"]))



labels_path = 'C:\Users\DINESH HEMANTH\Downloads\SOCIAL_DISTANCE_DETECTOR\coco.names'

labels = open(labels_path).read().strip().split("\n")

# load YOLO
net = cv2.dnn.readNet("C:\\Users\\DINESH HEMANTH\\Downloads\\SOCIAL_DISTANCE_DETECTOR\\yolov3.cfg", "C:\\Users\\DINESH HEMANTH\\Downloads\\SOCIAL_DISTANCE_DETECTOR\\yolov3.weights")

# getting the layers form network
ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

# loading video
cap = cv2.VideoCapture(args["input"] if args["input"] else 0)
#cap = cv2.VideoCapture("pedestrians.mp4")
out = None

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    results = detectPeople(frame, net, ln,)
    unsafe_persons = set()
    if len(results) >= 2:
        # extract the centroids and find the euclidian distance between 2
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in range(i+1, D.shape[1]):
                if D[i, j] < 60:
                    unsafe_persons.add(i)
                    unsafe_persons.add(j)
    for (i, (prob, box, centroid)) in enumerate(results):
        (sX, sY, eX, eY) = box
        (cX, cY) = centroid
        color = (0,255,0)
        if i in unsafe_persons:
            color = (0,0,255)
        # draw rectangle
        cv2.rectangle(frame, (sX, sY), (eX, eY), color, 2)
        cv2.circle(frame, (cX, cY), 2, (0,0,255), 2)
        text = "Social Distancing Violations: {}".format(len(unsafe_persons))
        cv2.putText(frame, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 3)

    if args["display"] > 0:
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if args["output"] != "" and out is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(args["output"], fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

    if out is not None:
        out.write(frame)

cap.release()
cv2.destroyAllWindows()




