# this code will be removed and I'll upload the new version with phisics
import bpy
from mathutils import Vector
import time
import math
import cv2
import mediapipe as mp
from statistics import mean
# start video stream
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0 

obj = bpy.data.objects["car"] # object that represents your direction
class car:
    def __init__(self,object):
        self.object = object
        #set the location
        self.object.location = (0,0,10)
        #set the rotation
        self.object.rotation_euler = (0,0,0)
        #display for some reason not having this causes disparity and the code take it's values from the unsynked dis
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        #bpy.ops.wm.redraw_timer(type='ANIM_STEP', iterations=10)
    def move(self,dis):
        a = self.object.rotation_euler
        #setting the new possitions off angle and distance/speed
        self.object.location[2] += (math.sin(a[0]) * math.cos(a[2]))*dis
        self.object.location[0] -= math.sin(a[2]) *dis
        self.object.location[1] += math.cos(a[0]) * math.cos(a[2]) *dis
        #refreshing
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
#        #bpy.ops.wm.redraw_timer(type='ANIM_STEP', iterations=10)
    def rotate(self,angle,value):
        self.object.rotation_euler[angle] = value
        #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    def turn(self,angle,value):
        self.object.rotation_euler[angle] += value
        #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
sedan = car(obj)
while True:
    pas, img = cap.read()
    imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        points = []
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img,(cx, cy),7,(255,0,255),cv2.FILLED)
                # cordinates r from 0-1 need to * them by w & h to get in function of screen
                cv2.putText(img,str(id),(cx, cy),cv2.FONT_HERSHEY_PLAIN,1, (255,0,0),2)
                if id == 5:
                    #index knucle, the two points where i'll base the steering angle off of 
                    points.append(lm)
            mpDraw.draw_landmarks(img,handLms, mpHands.HAND_CONNECTIONS)
        if len(points) > 1:
            # getting the ponit's x & y val
            cx, cy = int(points[0].x*w), int(points[0].y*h)
            cx1, cy1 = int(points[1].x*w), int(points[1].y*h)
            # finding the differance in x and y
            run = (cx - cx1)
            rise = (cy - cy1)
            # to prevent / by 0
            if run != 0:
                # turning in in rad and to go from ris/run we need to take the tan^-1
                sedan.turn(2,math.atan(rise/run)/2)
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                #moving forwards need to be done with a -
                sedan.move(-15)
                # give the user an idea of the angle at which they r truning in degrees 
                cv2.putText(img,str(math.degrees(math.atan(rise/run))),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
                print(math.atan(rise/run))
    # frames per second but i'm not currentty displaying it
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    k = cv2.waitKey(1)
    if k == 27:         #kill switch
        cv2.destroyAllWindows()
        break
#    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
    #show the user an image of themselves with angle and points on their hands
    cv2.imshow("image",img)
