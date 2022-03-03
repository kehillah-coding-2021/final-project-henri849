import cv2
import mediapipe as mp
import time
from statistics import mean
import tensorflow as tf
import os
import json
import numpy as np
import math

tf.config.set_soft_device_placement(True)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 820)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 440)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
#model =  tf.keras.models.load_model("models2")

with tf.io.gfile.GFile("./frozen_models/frozen_graph.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

# Wrap frozen graph to ConcreteFunctions
model = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=True)
sTime = 0
#model = tf.saved_model.load("models")
#min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity = 0
with mpHands.Hands() as hands:
    mpDraw = mp.solutions.drawing_utils
    while True:
        sTime = time.time()
        pas, img = cap.read()
        # scale_percent = 60 # percent of original size
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #img = cv2.resize(img,(320,240),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #img.flags.writeable = False
        #imageRGB = cv2.resize(imageRGB, (len(imageRGB)/2,len(imageRGB[0])/2), interpolation = cv2.INTER_AREA)
        results = hands.process(imageRGB)

        #img.flags.writeable = True
        if results.multi_hand_landmarks:
            print(len(results.multi_hand_landmarks))
            points = []
            z = 0
            if len(results.multi_hand_landmarks) > 1:
                h,w,c = img.shape
                p1 = [results.multi_hand_landmarks[0].landmark[9].x*w,results.multi_hand_landmarks[0].landmark[9].y*h]
                p2 = [results.multi_hand_landmarks[1].landmark[9].x*w,results.multi_hand_landmarks[1].landmark[9].y*h]
                c = (int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2))
                #c = (0,0)
                r = int(abs(c[0]-p1[0]))
                #wheel = cv2.imread('wheel.png')
                cv2.circle(img,c,r,(0,0,255),10)
            for handLms in results.multi_hand_landmarks:
                for id,lm in enumerate(handLms.landmark):
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    cv2.circle(img,(cx, cy),7,(255,0,255),cv2.FILLED)
                    # Les coordonnées sont de 0-1, pour les avoir en pixel par w et h
                    cv2.putText(img,str(id),(cx, cy),cv2.FONT_HERSHEY_PLAIN,1, (255,0,0),2)
                    if id == 5:
                        points.append(lm)
                hand = []
                start_point = 0
                end_point = 0
                for id,lm in enumerate(handLms.landmark):
                    if id == 0:
                        start_point = (int(lm.x*w),int(lm.y*h))
                    if id == 8:
                        end_point = (int(lm.x*w),int(lm.y*h))
                    if results.multi_handedness[z].classification[-1].label == "Right":
                        #ouput will be [hand,x,y,z]
                        hand += [float(1),lm.x,lm.y,int(lm.z*100)/100]
                    else:
                        # left
                        hand += [float(0),lm.x,lm.y,int(lm.z*100)/100]
                #print(hand)
                hand += [0] * (252 - len(hand))
                hand = np.array([hand,])
                hand = tf.ragged.constant(hand, dtype=tf.float32).to_tensor()
                fasterRes = model(x=hand)[0][0]
                #print(fasterRes[0][0])
                #fasterRes = model.predict(hand)[0]
                if len(points) > 1:
                    cx, cy = int(points[0].x*w), int(points[0].y*h)
                    cx1, cy1 = int(points[1].x*w), int(points[1].y*h)
                    # find the rise over run to that we can find the slope to then find the angle
                    run = (cx - cx1)
                    rise = (cy - cy1)
                    cv2.line(img, (cx,cy), (cx1,cy1), (0,0,255), 2)
                    if run == 0:
                        run = 0.0001
                    cv2.putText(img,str(int(math.atan(rise/run)*180/math.pi)),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (0,0,255),3)
                
                if fasterRes[0] > fasterRes[1]:
                    print("faster",fasterRes,start_point)
                    img = cv2.rectangle(img, start_point, end_point, (0,255,0), 4)
                else:
                    print("no acseleration",fasterRes)
                print(fasterRes)
                
                mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)
                z +=1

        fps = 1/(time.time()-sTime)
        k = cv2.waitKey(30)
        if k == 27:         #pour quiter le program
            cv2.destroyAllWindows()
            break
        cv2.putText(img,str(int(fps)),(90,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
        cv2.imshow("image",img)
        cv2.waitKey(1)

cap.release()
