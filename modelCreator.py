import cv2
import mediapipe as mp
import time
from statistics import mean
import json
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0 
while True:
    pas, img = cap.read()
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(imageRGB)
    # If we got results
    if results.multi_hand_landmarks:
      #loop through both hands
        for handLms in results.multi_hand_landmarks:
            # seperate ids and positions
            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                # the cordinates are from 0-1 so multipling by fram l or w gives it in function of the frame
                cx, cy = int(lm.x*w), int(lm.y*h)
                #displaying the provided points
                cv2.circle(img,(cx, cy),7,(255,0,255),cv2.FILLED)
                # with id label
                cv2.putText(img,str(id),(cx, cy),cv2.FONT_HERSHEY_PLAIN,1, (255,0,0),2)
        # where we store formated points
        x = []
        #looping through the points and converting them from objects to arrays (we will be stringifing them and JSON can't take objects)
        for i in results.multi_hand_landmarks:
            x.append([])
            for id,lm in enumerate(i.landmark):
                x[-1].append([id,[lm.x,lm.y,lm.z]])
        #saving steering frames
        if cv2.waitKey(33) == ord('w'):         #pour sauvegarder un frame
            page = open("wheel.txt","a")
            page.write("\n\n\n\n ~||~" +json.dumps(x) )
            page.close()
        # saing steering + acseletration frames
        if cv2.waitKey(33) == ord('a'):         #pour sauvegarder un frame
            page = open("wheel+acse.txt","a")
            page.write("\n\n\n\n ~||~" +json.dumps(x) )
            page.close()
    # adding an fps display
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #kill the problem switch(esc)
    k = cv2.waitKey(1)
    if k == 27:         #pour quiter le program
        cv2.destroyAllWindows()
        break
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
    cv2.imshow("image",img)
    cv2.waitKey(1)
