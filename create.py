#bge is Blender game engine, I use it to set gloabal variables
import bge
#import tensorflow to use my model and set a global variable with it as it;s value
import tensorflow as tensorflow
bge.logic.tensorflow = tensorflow
#import opencv to use the video capture function and set a global variable with it as it;s value
import cv2 as cv2
bge.logic.cv2 = cv2
#import mediapipe to use their hadn model and set a global variable with it as it;s value
import mediapipe as mp
bge.logic.mp = mp
#importing math, it's light but it saves processing power putting it into a global variable
import math
bge.logic.math = math
#numpy
import numpy as np
bge.logic.np = np
#setting a globval variable as my model, THIS SAVES A LOT OF SPEED AND IS NESSISARY
bge.logic.model =  bge.logic.tensorflow.keras.models.load_model("models")
#inits the video capture 
bge.logic.cap = bge.logic.cv2.VideoCapture(0, bge.logic.cv2.CAP_DSHOW)  
#the mediapipe hand lib
bge.logic.mpHands = mp.solutions.hands
bge.logic.hands = bge.logic.mpHands.Hands()
#and sets the hands up
bge.logic.mpDraw = bge.logic.mp.solutions.drawing_utils 
