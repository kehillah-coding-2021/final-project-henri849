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
            start_point = 0
            end_point = 0
            for id,lm in enumerate(handLms.landmark):
                # taking two points to make box around the hand
                if id == 0:
                    start_point = (int(lm.x*w),int(lm.y*h))
                if id == 8:
                    end_point = (int(lm.x*w),int(lm.y*h))
                    # if it's a right hand
                if results.multi_handedness[z].classification[-1].label == "Right":
                    #ouput will be [hand,x,y,z]
                    hand += [float(1),lm.x,lm.y,int(lm.z*100)/100]
                else:
                    # left
                    hand += [float(0),lm.x,lm.y,int(lm.z*100)/100]
            #print(hand)
            # format it into an array 3 times it's size (when making the data mediapipe got confused and sometimes thought I was showing three hands and the input layer of the model must have a fix size)
            hand += [0] * (252 - len(hand))
            # convert it into an np array (so that I can convert it into a ragged tensor (the model can only take tensors))
            hand = np.array([hand,])
            # convert to ragged tensor then to normal tensor (I didn't figure out a way to go dirrectly to a tensor) 
            hand = tf.ragged.constant(hand).to_tensor()
            # inpout the data into the model
            fasterRes = model.predict(hand)[0]
            # if the first neuron (acselerate) is more acttivated then the other
            if fasterRes[0] > fasterRes[1]:
                # inform the user
                print("faster",fasterRes,start_point)
                # add a rectange around the hand for visual feedback
                img = cv2.rectangle(img, start_point, end_point, (0,255,0), 4)
            else:
                # inform the user of the negative output
                print("no acseleration",fasterRes)
            #print(model.predict(hand))
            #draw points on hand and conenctiosn
            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)
        z +=1
    # check for the user pressing on escape
    k = cv2.waitKey(1)
    if k == 27:         # if so kill the program
        cv2.destroyAllWindows()
        break
       # show the user themselves, the points/connections and whether it thinks they are accelerating 
    cv2.imshow("image",img)
    cv2.waitKey(1)
