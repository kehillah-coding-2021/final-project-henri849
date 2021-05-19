import cv2
import mediapipe as mp
import time
from statistics import mean
import json
# start vid capture
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0 
j = 0
k = 0
while True:
    pas, img = cap.read()
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img,(cx, cy),7,(255,0,255),cv2.FILLED)
                # Les coordonnées sont de 0-1, pour les avoir en pixel par w et h
                cv2.putText(img,str(id),(cx, cy),cv2.FONT_HERSHEY_PLAIN,1, (255,0,0),2)
            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)
        x = []
        z = 0
        for i in results.multi_hand_landmarks:
            x.append([])
            for id,lm in enumerate(i.landmark):
                #print(results.multi_handedness[z].classification[-1].label)
                x[-1].append([[id,results.multi_handedness[z].classification[-1].label],[lm.x,lm.y,lm.z]])
            z += 1 
        l  = cv2.waitKey(1)
        if l == ord('a'):         #goods
            page = open("wheel+acse.txt","a")
            j +=1
            print("pass",j)
            page.write("\n"+json.dumps(x) )
            page.close()
        if l == ord('d'):         #bads
            k += 1
            print("bad",k)
            page = open("bads.txt","a")
            page.write("\n"+json.dumps(x) )
            page.close()
        if l == 27:         #quit
            cv2.destroyAllWindows()
            break
    # telling the user the current refresh frames per second
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
    #display them with the points on their hand
    cv2.imshow("image",img)
    cv2.waitKey(1)
