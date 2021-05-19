import tensorflow as tf
import json
import numpy as np
ans = open("output.txt","r")
ans = json.loads(ans.read())

[[x_train, x_test], [y_train, y_test]] = ans
#print(len(x_train[0]),"YYYYYYYYYYYYYYY")
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Input(shape=(4,)),
#   tf.keras.layers.Dense(48, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(2, activation='softmax')
# ])
# x_train=np.asarray(x_train).astype(np.float32)
# x_test=np.asarray(x_test).astype(np.float32)
# tf.convert_to_tensor(x_train)
# tf.convert_to_tensor(x_test)
# tf.convert_to_tensor(y_train)
# tf.convert_to_tensor(y_test)

x_train = tf.ragged.constant(x_train).to_tensor()
x_test = tf.ragged.constant(x_test).to_tensor()
y_train = tf.ragged.constant(y_train).to_tensor()
y_test = tf.ragged.constant(y_test).to_tensor()
print(x_train)
model = tf.keras.Sequential([
    tf.keras.layers.Input(252,),
#     tf.keras.layers.Input(252),
    tf.keras.layers.Dense(48, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20)
model.evaluate(x_test,  y_test, verbose=2)
model.save("models")