import cv2
import mediapipe as mp
import time
from statistics import mean
import tensorflow as tf
import json
import numpy as np
import math
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
#model = tf.saved_model.load("models")
model =  tf.keras.models.load_model("models")
while True:
    pas, img = cap.read()
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        points = []
        z = 0
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
            for id,lm in enumerate(handLms.landmark):
                if results.multi_handedness[z].classification[-1].label == "Right":
                    #ouput will be [hand,x,y,z]
                    hand += [float(1),lm.x,lm.y,int(lm.z*100)/100]
                else:
                    # left
                    hand += [float(0),lm.x,lm.y,int(lm.z*100)/100]
            #print(hand)
            hand += [0] * (252 - len(hand))
            hand = np.array([hand,])
            hand = tf.ragged.constant(hand).to_tensor()
            fasterRes = model.predict(hand)[0]
            if fasterRes[0] > fasterRes[1]:
                print("faster",fasterRes)
            else:
                print("no acseleration",fasterRes)
            #print(model.predict(hand))
            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)
        z +=1
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    k = cv2.waitKey(1)
    if k == 27:         #pour quiter le program
        cv2.destroyAllWindows()
        break
#    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
    cv2.imshow("image",img)
    cv2.waitKey(1)