**Panda3d racing game + mediapipe + other models to make it actually controllable**

**WHAT IT'S SUPPOSED TO DO:
Most human computer interface devices (such as mice, keyboards, joysticks, and game controllers) require a certain level of dexterity and can be challenging for people with certain disabilities. The project goal is to enhance computer accessibility for those people without increasing cost. In the first phase, a recreational activity is made accessible to all by utilizing the computer webcam and AI as an input device for an automobile racing game, thus eliminating the need for keyboard, mouse, or external game controllers.**


The Panda 3D branch is the 2nd iteration as the Blender Game Engine was obsoleat and didn't had perfeormances issues especialy when exporting the aplication.

The stucture of it all is pretty simple, Inside of 

**CVmodels (The folder responsible for creating the gesture recognition models):**

.....mediapipetest1.py -> incharge of data colection, runs mediapipe and when the user presses w,a,d the gestures are converted to json and writen to different files (based on  letter label)

.....dataformat.py -> formats the data into two lists [data,labels] and shuffles the data 

.....modelcreator.py -> creates the model and then freezes it

.....Tester.py -> allows user to test the created models
  
  
**Game code (Stored in Panda folder)**

.....Car.py -> Handels pretty much everything, from loading assets, gesture models, creating game objects, bounding boxes, game variables, ect

.....models/HeightField2.png -> default heighfeild image used as terain (I am working on a version that acsepts models as terains)


**Conda yml import** rather than installing all the dependencies and all(which are listed in the requirements.txt), if you use conda you can simply use the provided yml file
Here's the conda docs on importing them:

.....https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file

**Possible errors and fixes:**

  tensorflow.python.framework.errors_impl.NotFoundError: NewRandomAccessFile failed to Create/Open: frozen_models/frozen_graph.pb : The system cannot find the path specified.    
  ; No such process

  **Easiest fix** -> *run from command line*, this seems to be a problem related to where the file is being executed from, *I am working on a fix*


