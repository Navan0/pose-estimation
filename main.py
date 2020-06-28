import cv2 as cv
import numpy as np
from operator import mul
import math


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = 368
inHeight = 368
cframe = 0

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

# cap = cv.VideoCapture(0)
# cap = cv.VideoCapture('parkinsongait.mp4')
cap = cv.VideoCapture('hemiplegic.mp4')

park = []
hemo = []
cet = []
cetf = False
while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    # import pdb; pdb.set_trace()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

    assert(len(BODY_PARTS) == out.shape[1])

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > 0.1 else None)
            # print(points[13])
        # if len(points) >= 14:
            # print("R ankle"+str(points[13]),"L ankle"+str(points[10]))
            # print(list(BODY_PARTS.keys().index(1)))
            # import pdb; pdb.set_trace()
        parts = list(BODY_PARTS.keys())
        part = list(BODY_PARTS.values())

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        # print(BODY_PARTS[partFrom],BODY_PARTS[partTo])

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.putText(frame,parts[part.index(9)], points[9], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2, cv.LINE_AA)
            cv.putText(frame,parts[part.index(10)], points[10], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2, cv.LINE_AA)
            cv.putText(frame,parts[part.index(12)], points[12], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2, cv.LINE_AA)
            cv.putText(frame,parts[part.index(13)], points[13], cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2, cv.LINE_AA)

    # import pdb; pdb.set_trace()
    print(points[2][0])

    try:

        if abs(points[13][0]) - abs(points[10][0]) < 50 and abs(points[12][0]) - abs(points[9][0]) < 50:
            park.append(abs(points[13][0]) - abs(points[10][0]))
        # cv.putText(frame, str("Parkison gait"), (100,200), cv.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)
        elif abs(points[13][0]) - abs(points[10][0]) > 50 and abs(points[12][0]) - abs(points[9][0]) > 50 and abs(points[2][0]) > 600:
            hemo.append(abs(points[13][0]) - abs(points[10][0]))
        elif abs(points[2][0]) < 600 :
            cet.append("1")
    except:
        cv.putText(frame, str("finding"), (100,200), cv.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 5)
    print(cet)

    cframe += 1

    if len(cet) > 8:
        cetf = True

    if len(hemo) > 120 or len(park) > 120 or cetf is True:
        # if max(len())
        # import pdb; pdb.set_trace()

        if len(hemo) > len(park) and cetf is False:
            cv.putText(frame, str("Hemiplegic Gait"), (100,200), cv.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)
        elif len(hemo) < len(park) and cetf is False:
            cv.putText(frame, str("Parkison gait"), (100,200), cv.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)
        elif cetf is True:
            cv.putText(frame, str("myopathic gait"), (100,200), cv.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)



        # cv.putText(frame, str("Hemiplegic Gait"), (100,200), cv.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)
    # elif abs(points[13][0]) - abs(points[10][0]) > 50 and abs(points[12][0]) - abs(points[9][0]) > 20 > 40:
    #     cv.putText(frame, str("Hemiplegic Gait"), (100,200), cv.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)




    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    # cv.putText(frame,ankleDist, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2, cv.LINE_AA)

    cv.imshow('OpenPose using OpenCV', frame)


# cv.putText(frame, str("Hemiplegic Gait"), (100,200), cv.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)
