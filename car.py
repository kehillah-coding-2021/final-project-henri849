import bpy
from mathutils import Vector
import time
import math

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
        #bpy.ops.wm.redraw_timer(type='ANIM_STEP', iterations=10)
    def rotate(self,angle,value):
        self.object.rotation_euler[angle] = value
