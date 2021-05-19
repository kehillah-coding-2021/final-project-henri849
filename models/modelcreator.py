import tensorflow as tf
import json
import numpy as np
# open data
ans = open("output.txt","r")
# make it python
ans = json.loads(ans.read())
# split it into test train, labels and data
[[x_train, x_test], [y_train, y_test]] = ans
# beacause tensoflow is painful and doesn't let you do anything I need to spend hours trying to find a dumb fix, this makes it work
x_train = tf.ragged.constant(x_train).to_tensor()
x_test = tf.ragged.constant(x_test).to_tensor()
y_train = tf.ragged.constant(y_train).to_tensor()
y_test = tf.ragged.constant(y_test).to_tensor()
#create the model
model = tf.keras.Sequential([
    # why 252 because tensoflow said so, well when changing an array to ragged tesnflow then to a normal tensor causes deformation
    # created an 252 neuron input layer
    tf.keras.layers.Input(252,),
    # 48 neuron "deep" layer
    tf.keras.layers.Dense(48, activation='relu'),
    # ouput layer, softmax makes you feel better about the resuluts, well it actualy round percetages eg: [0.6,0.4] == [1,0] and just makes the final score higher 
    tf.keras.layers.Dense(2, activation='softmax')
])
# compile, categorical_crossentropy bc unlike sparse_categorical_crossentropy we are dealing with not 1d tensors
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# train the model for 20 epochs with a bach size of about 135, it achives about 94.92% acuracy on my over 4300 training dataset
model.fit(x_train, y_train, epochs=20)
# test it on the test datasets, it achives ~96.45
model.evaluate(x_test,  y_test, verbose=2)
# save the model into the models folder
model.save("models")
