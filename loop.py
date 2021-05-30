import bge
from bge import logic
math = bge.logic.math
# this is being run inside the car's logic so the current controler is the car
cont = logic.getCurrentController()
#getting the car object
own = cont.owner
#car player that will keep track of rotatio and forces
class player:
    def __init__(self,obj):
        self.a = obj.localOrientation.to_euler()
        self.own = obj
        self.location = obj.position
    def calcloc(self,dis=1):
        self.location = self.own.position
        self.a = self.own.localOrientation.to_euler()
        a =  self.a
        #updating forces off rotation and speed
        self.location[2] += (math.sin(a[0]) * math.cos(a[2]))*dis
        self.location[0] += math.sin(a[2]) *dis
        self.location[1] -= math.cos(a[0]) * math.cos(a[2]) *dis
# reading the webcam strea,
pas, img = bge.logic.cap.read()
#converting to BGR, I accidentally did this durring training and now I need it
imageRGB = bge.logic.cv2.cvtColor(img,bge.logic.cv2.COLOR_BGR2RGB)
# run the stream through mediapipe
results = bge.logic.hands.process(imageRGB)
#if there are hand to anasile
if results.multi_hand_landmarks:
    #were we keep track of the two knuckles for steering angle
    points = []
    # we need z to keep track of what hand we are on
    z  = 0
    #fasterRes is were we put the model results and the box for the visual feedback
    fasterRes = []
    #looping through each hand
    for handLms in results.multi_hand_landmarks:
       # han is where we will fromat all the points to run it into my gesture analisis model 
        hand = []
        #looping through each point on each hand
        for id,lm in enumerate(handLms.landmark):
              # getting screen dimentions
              h,w,c = img.shape
              # finding point locations bassed of what media pipe return streactch onto our screen
              cx, cy = int(lm.x*w), int(lm.y*h)
              #add a circle on top of each point
              bge.logic.cv2.circle(img,(cx, cy),7,(255,0,255),bge.logic.cv2.FILLED)
              # and it's id
              bge.logic.cv2.putText(img,str(id),(cx, cy),bge.logic.cv2.FONT_HERSHEY_PLAIN,1, (255,0,0),2)
              # if it's the knuckles are stored (index knuckles I think)
              if id == 5:
                  points.append(lm)
              #then keeping two points to make a box around the hand
              if id == 0:
                  start_point = (int(lm.x*w),int(lm.y*h))
              if id == 8:
                  end_point = (int(lm.x*w),int(lm.y*h))
              #formating data so that the model can read it and rounding z bc it somtimes has a ridiculous amount of decimal points
              if results.multi_handedness[z].classification[-1].label == "Right":
                  #ouput will be [hand,x,y,z]
                  hand += [float(1),lm.x,lm.y,int(lm.z*100)/100]
              else:
                  # left
                  hand += [float(0),lm.x,lm.y,int(lm.z*100)/100]
        # put it into an array 3 times it's size (I trained it on multiple hands, 3 at most and the input layer must have a fixed size)
        hand += [0] * (252 - len(hand))
        #convert to an np array so that I can convert it into a ragged tensor
        hand = bge.logic.np.array([hand,])
        #convert the ragged tensor into a normal tensor so tensorflow can use it
        hand = bge.logic.tensorflow.ragged.constant(hand).to_tensor()
        #put the computed result into our list for latter desision making
        fasterRes.append([bge.logic.model.predict(hand)[0],[start_point,end_point]])
        z += 1
        #draw hand connections
        bge.logic.mpDraw.draw_landmarks(img,handLms, bge.logic.mpHands.HAND_CONNECTIONS)
    # if we have two hands
    if len(results.multi_hand_landmarks) > 1:
        # find the location of the saved box points in terms of the screen size
        cx, cy = int(points[0].x*w), int(points[0].y*h)
        cx1, cy1 = int(points[1].x*w), int(points[1].y*h)
        # find the rise over run to that we can find the slope to then find the angle
        run = (cx - cx1)
        rise = (cy - cy1)
        # to not divide by 0
        if run != 0:
            #acsses car
            sedan = player(own)
            #rotate it with the angle of the knuckles
            own.applyRotation([0,0,math.atan(rise/run)/4.5])
            # looping through the saved data
            for each in fasterRes:
                #if the model thinks were accelerating
                if each[0][0] > each[0][1]:
                    #calculator force 
                    sedan.calcloc(0.5)
                    # update object (yes it's unituitive but it's all that works)
                    own.applyForce((0,0,0))
                    # visual feedback a green rectable around the hand
                    img = bge.logic.cv2.rectangle(img, each[1][0], each[1][1], (0,255,0), 4)
                #tell the user their turning angle
                bge.logic.cv2.putText(img,str(math.degrees(math.atan(rise/run))),(10,70),bge.logic.cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
#show them themselves with the added visual information
bge.logic.cv2.imshow("image",img)
