import sys
from xml.etree.ElementTree import PI
import direct.directbase.DirectStart

from direct.showbase.DirectObject import DirectObject
from direct.showbase.InputStateGlobal import inputState

from panda3d.core import AmbientLight, load_prc_file,DirectionalLight,Vec3,Vec4,LPoint3f,Point3,TransformState,BitMask32,GeoMipTerrain,Filename,PNMImage,Texture, CardMaker

from panda3d.bullet import BulletWorld,ZUp,BulletPlaneShape,BulletHeightfieldShape,BulletBoxShape,BulletRigidBodyNode,BulletDebugNode,BulletVehicle
from direct.gui.OnscreenImage import OnscreenImage
from direct.gui.OnscreenText import OnscreenText
import cv2
import mediapipe as mp
import time
from statistics import mean
import tensorflow as tf
import json
import numpy as np
import math
import pathlib
#load_prc_file('config.prc')
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

with tf.io.gfile.GFile(dir_path+ "/frozen_models/frozen_graph.pb", "rb") as f:
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




class Game(DirectObject):

	def __init__(self):
		base.setBackgroundColor(0.1, 0.1, 0.8, 1)
		base.setFrameRateMeter(True)
		self.cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
		base.cam.setPos(0, -20, 4)
		base.cam.lookAt(0, 0, 0)
		#self.steer = mediaLogic()
		# Light
		alight = AmbientLight('ambientLight')
		alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
		alightNP = render.attachNewNode(alight)

		dlight = DirectionalLight('directionalLight')
		dlight.setDirection(Vec3(1, 1, -1))
		dlight.setColor(Vec4(0.7, 0.7, 0.7, 1))
		dlightNP = render.attachNewNode(dlight)
		self.delta =0
		self.blocks = []

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands()
		#self.model =  tf.keras.models.load_model("models")
		self.model = wrap_frozen_graph(graph_def=graph_def,
									inputs=["x:0"],
									outputs=["Identity:0"],
									print_graph=True)


		self.height = 20.0
		img = PNMImage(Filename(pathlib.PurePosixPath(pathlib.Path(dir_path+'\models\HeightField2.png'))))
		offset = img.getXSize() / 2.0 -0.5
		self.terrain = GeoMipTerrain("terrain")
		self.terrain.setHeightfield(Filename( pathlib.PurePosixPath(pathlib.Path(dir_path+"\models\HeightField2.png"))))

		self.root = self.terrain.getRoot()
		self.root.reparentTo(render)
		self.root.setSx(0.875)
		self.root.setSy(0.875)
		self.root.setSz(self.height)
		self.root.setPos(-offset, -offset, -self.height / 2.0)
		render.clearLight()
		render.setLight(alightNP)
		render.setLight(dlightNP)



		# Input
		self.accept('escape', self.doExit)
		self.accept('r', self.doReset)
		self.accept('f1', self.toggleWireframe)
		self.accept('f2', self.toggleTexture)
		self.accept('f3', self.toggleDebug)
		self.accept('f5', self.doScreenshot)

		inputState.watchWithModifiers('forward', 'w')
		inputState.watchWithModifiers('left', 'a')
		inputState.watchWithModifiers('reverse', 's')
		inputState.watchWithModifiers('right', 'd')
		inputState.watchWithModifiers('turnLeft', 'q')
		inputState.watchWithModifiers('turnRight', 'e')

		# Task
		taskMgr.add(self.update, 'updateWorld')

		# Physics
		self.setup()

  # _____HANDLER_____

	def doExit(self):
		self.cleanup()
		sys.exit(1)

	def doReset(self):
		self.cleanup()
		self.setup()

	def toggleWireframe(self):
		base.toggleWireframe()

	def toggleTexture(self):
		base.toggleTexture()

	def toggleDebug(self):
		if self.debugNP.isHidden():
			self.debugNP.show()
		else:
			self.debugNP.hide()

	def doScreenshot(self):
		base.screenshot('Bullet')

  # ____TASK___

	def processInput(self, dt, steeringIncrement):
		engineForce = 0.0
		brakeForce = 0.0

		if inputState.isSet('forward'):
			engineForce = 700.0
			brakeForce = 0.0

		if inputState.isSet('reverse'):
			engineForce = 0.0
			brakeForce = 100.0

		if inputState.isSet('turnLeft'):
			self.steering += dt * steeringIncrement
			self.steering = min(self.steering, self.steeringClamp)
			# Apply steering to front wheels
			self.vehicle.setSteeringValue(self.steering, 0)
			self.vehicle.setSteeringValue(self.steering, 1)

		if inputState.isSet('turnRight'):
			self.steering -= dt * steeringIncrement
			self.steering = max(self.steering, -self.steeringClamp)
			# Apply steering to front wheels
			self.vehicle.setSteeringValue(self.steering, 0)
			self.vehicle.setSteeringValue(self.steering, 1)
			
			# Apply engine and brake to rear wheels
			self.vehicle.applyEngineForce(engineForce, 2)
			self.vehicle.applyEngineForce(engineForce, 3)
			self.vehicle.setBrake(brakeForce, 2)
			self.vehicle.setBrake(brakeForce, 3)

	def update(self, task):
		for i in self.blocks:
			i.destroy()
		dt = globalClock.getDt()
		self.terrain.update()
		pas, img = self.cap.read()
		d = self.logic(img)
		inputState.set("forward",False)
		inputState.set("reverse",False)
		inputState.set("turnLeft",False)
		inputState.set("turnRight",False)
		if d[0] != -1:
			#print(d[0])
			if d[0]:
				inputState.set("forward",True)
			else:
				inputState.set("reverse",True)
			#print(inputState.isSet('forward'))
			if d[1] != 0:
				if d[1]*180/math.pi > 0:
					inputState.set("turnLeft",True)
				else:
					inputState.set("turnRight",True)
			#print(d[1]*180/math.pi/90)#*math.cos(d[1])
			self.processInput(dt,abs(d[1]*180/math.pi/90)*20)#*(180/math.pi)
		else:
			self.processInput(dt,self.steeringIncrement)
			if self.vehicle.getSteeringValue(0) > 0:
				self.steering -=abs(self.steering/10)
			else:
				self.steering +=abs(self.steering/10)
		if int(dt*100 % 2) ==0:
			base.cam.setPos(base.cam.getPos() +0.003) 
			self.delta +=0.003
		else:
			base.cam.setPos(base.cam.getPos() -0.003) 
			self.delta -=0.003
		if abs(self.delta) > 0.02:
			print("CLEaR")
			base.cam.setPos(base.cam.getPos() -self.delta) 
			self.delta = 0
		print(self.delta)
		#friction, kinda having wheel drift to center https://discourse.panda3d.org/t/springy-camera/6069
		# cameraTargetPos = self.ideal.getPos(render) # point behind the aircraft
		# cameraCurrentPos = base.cam.getPos(render)
		# cameraMoveVec = cameraTargetPos - cameraCurrentPos # or current - target?
		# base.cam.lookAt(cameraTargetPos)

		# # 0.1 is the factor how fast the camera follows this point, it's depending on your taste
		# base.cam.setPos(base.cam, cameraMoveVec * dt * 0.2) 
		#print(self.vehicle.getWheel(4).getChassisConnectionPointCs())
		# base.cam.reparentTo(self.vehicle.getWheel(4))
		#base.cam.setPos(self.vehicle.getWheel(4).getChassisConnectionPointCs())
		#print(self.vehicle.getSteeringValue(0),self.vehicle.getSteeringValue(1),self.steering)
		# else:
		
		#print(self.vehicle.getSteeringValue(0),self.vehicle.getSteeringValue(1),inputState.isSet('turnLeft'),inputState.isSet('turnRight'),2)
		
		# print("HERE ",self.vehicle.getWheel(0).getSuspensionRelativeVelocity(),base.cam.getPos())#base.cam.getPos()
		# pos = base.cam.getPos()
		# pos[2] += self.vehicle.getWheel(0).getSuspensionRelativeVelocity()
		# if pos[2] < 0:
		# 	pos[2] = 1.7
		# base.cam.setPos(pos) 
		#base.cam.lookAt(self.vehicle.getChassis.getPos())
		#self.processInput(dt,self.steeringIncrement)
		self.world.doPhysics(dt, 10, 0.008)
		#print(self.vehicle.get_forward_vector())
		#+ self.vehicle.get_forward_vector().getImplicitVelocity()
		#textObject = OnscreenText(text='speed = ', pos=(-0.5, 0.02), scale=0.07)
		#print self.vehicle.getWheel(0).getRaycastInfo().isInContact()
		#print self.vehicle.getWheel(0).getRaycastInfo().getContactPointWs()

		#print self.vehicle.getChassis().isKinematic()

		return task.cont

	def logic(self, img):
		sTime = 0
		#print("run")
		blocks = []
		h,w,c = img.shape
		with self.mpHands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5, model_complexity = 1) as hands:
			#mpDraw = mp.solutions.drawing_utils
			# sTime = time.time()
			#img = cv2.resize(img,(320,240),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
			imageRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
			imageRGB.flags.writeable = False

			results = hands.process(imageRGB)

			imageRGB.flags.writeable = True
			if results.multi_hand_landmarks:
				#print(len(results.multi_hand_landmarks))
				points = []
				z = 0
				#print(len(results.multi_hand_landmarks))
				if len(results.multi_hand_landmarks) > 1:
					p1 = [results.multi_hand_landmarks[0].landmark[9].x*w,results.multi_hand_landmarks[0].landmark[9].y*h]
					p2 = [results.multi_hand_landmarks[1].landmark[9].x*w,results.multi_hand_landmarks[1].landmark[9].y*h]
					c = (int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2))
					#c = (0,0)
					r = int(abs(c[0]-p1[0]))

					cv2.circle(img,c,r,(46,74,98),20)
				m = False
				rise = 0
				run = 0.001
				for handLms in results.multi_hand_landmarks:
					hand = []
					start_point = 0
					end_point = 0
					for id,lm in enumerate(handLms.landmark):
						if id == 5:
							points.append(lm)
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
					fasterRes = self.model(x=hand)[0][0]
					# hand = tf.ragged.constant(hand).to_tensor()
					#fasterRes = self.model.predict(hand)[0]
					if len(points) > 1:
						cx, cy = int(points[0].x*w), int(points[0].y*h)
						cx1, cy1 = int(points[1].x*w), int(points[1].y*h)
						# find the rise over run to that we can find the slope to then find the angle
						run = (cx - cx1)
						rise = (cy - cy1)
					if fasterRes[0] > fasterRes[1]:
						#k = cv2.waitKey(1)
						img = cv2.rectangle(img, start_point, end_point, (0,255,0), 4)
						m = True
						for id,lm in enumerate(handLms.landmark):
							h,w,c = img.shape
							cx, cy = int(lm.x*w), int(lm.y*h)
							cv2.circle(img,(cx, cy),7,(255,0,255),cv2.FILLED)
							# Les coordonnées sont de 0-1, pour les avoir en pixel par w et h
							cv2.putText(img,str(id),(cx, cy),cv2.FONT_HERSHEY_PLAIN,1, (255,0,0),2)
							#print(lm.x, lm.y,lm.z)
							v = OnscreenImage(image="models\green.png", pos=(0.5-lm.x,lm.z,-lm.y), scale=(0.01,0.01,0.01))
							self.blocks.append(v)
					else:
						for id,lm in enumerate(handLms.landmark):
							h,w,c = img.shape
							cx, cy = int(lm.x*w), int(lm.y*h)
							cv2.circle(img,(cx, cy),7,(255,0,255),cv2.FILLED)
							# Les coordonnées sont de 0-1, pour les avoir en pixel par w et h
							cv2.putText(img,str(id),(cx, cy),cv2.FONT_HERSHEY_PLAIN,1, (255,0,0),2)
							#print(lm.x, lm.y,lm.z)
							v = OnscreenImage(image="models\\red.png", pos=(0.5-lm.x,lm.z,-lm.y), scale=(0.01,0.01,0.01))
							self.blocks.append(v)

				img = img.astype('float32')
				img /= 255.0
				#print(img)
				if run ==0:
					run =0.0001
				cv2.imshow("image",img)
				return [m,math.atan(rise/run)]



			cv2.imshow("image",img)
			#self.cap.release()
			return [-1]
	def cleanup(self):
		self.world = None
		self.worldNP.removeNode()

	def setup(self):
		self.worldNP = render.attachNewNode('World')

		# World
		self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
		self.debugNP.show()

		self.world = BulletWorld()
		self.world.setGravity(Vec3(0, 0, -18.81))
		self.world.setDebugNode(self.debugNP.node())

		# Plane
		img = PNMImage(Filename(pathlib.PurePosixPath(pathlib.Path(dir_path+'\models\HeightField2.png'))))
		shape = BulletHeightfieldShape(img, self.height, ZUp)

		self.groundNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
		self.groundNP.node().addShape(shape)
		#self.groundNP.setPos(0, 0, 0)
		self.groundNP.setCollideMask(BitMask32.allOn())
		self.world.attachRigidBody(self.groundNP.node())

		# Chassis
		shape = BulletBoxShape(Vec3(0.6, 1.4, 0.5))
		ts = TransformState.makePos(Point3(0, 0, 0.5))

		np = self.worldNP.attachNewNode(BulletRigidBodyNode('Vehicle'))
		np.node().addShape(shape, ts)
		np.setPos(0, 0, 1)
		np.node().setMass(800.0)
		np.node().setDeactivationEnabled(False)

		self.world.attachRigidBody(np.node())

		#np.node().setCcdSweptSphereRadius(1.0)
		#np.node().setCcdMotionThreshold(1e-7) 

		# Vehicle
		self.vehicle = BulletVehicle(self.world, np.node())
		self.vehicle.setCoordinateSystem(ZUp)
		self.world.attachVehicle(self.vehicle)

		self.yugoNP = loader.loadModel('models/finalcar.gltf')
		self.yugoNP.reparentTo(np)

		# Right front wheel
		np = loader.loadModel('models/wheel2.gltf')
		np.reparentTo(self.worldNP)
		self.addWheel(Point3( 0.70,  1.05, 0.3), True, np)

		# Left front wheel
		np = loader.loadModel('models/wheel2.gltf')
		np.reparentTo(self.worldNP)
		self.addWheel(Point3(-0.70,  1.05, 0.3), True, np)

		# Right rear wheel
		np = loader.loadModel('models/wheel2.gltf')
		np.reparentTo(self.worldNP)
		self.addWheel(Point3( 0.70, -1.05, 0.3), False, np)

		# Left rear wheel
		np = loader.loadModel('models/wheel2.gltf')
		np.reparentTo(self.worldNP)
		self.addWheel(Point3(-0.70, -1.05, 0.3), False, np)

		# CAM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# self.ideal = loader.loadModel('models/box.egg')
		# self.ideal.setPos(0, -10.7, 3.7)
		# self.ideal.reparentTo(self.yugoNP)
		base.cam.reparentTo(self.yugoNP)#    base.cam.reparentTo(self.vehicle.getWheel(4))
		# base.cam.setPos(0, -1.7, 2)
		# base.cam.lookAt(0, 5, 2-1.5)
		base.cam.setPos(0, -1.7, 1.7)
		base.cam.lookAt(0, 5, 1.0)
		#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		# Steering info
		self.steering = 0.0            # degree
		self.steeringClamp = 10.0      # degree 45
		self.steeringIncrement = 120.0 # degree per second 120.0 normaly

	def addWheel(self, pos, front, np):
		wheel = self.vehicle.createWheel()

		wheel.setNode(np.node())
		wheel.setChassisConnectionPointCs(pos)
		wheel.setFrontWheel(front)

		wheel.setWheelDirectionCs(Vec3(0, 0, -1))
		wheel.setWheelAxleCs(Vec3(1, 0, 0))
		wheel.setWheelRadius(0.30)
		wheel.setMaxSuspensionTravelCm(40.0)

		wheel.setSuspensionStiffness(40.0)
		wheel.setWheelsDampingRelaxation(2.3)
		wheel.setWheelsDampingCompression(4.4)
		wheel.setFrictionSlip(300.0)
		wheel.setRollInfluence(0.1)
game = Game()
run()
