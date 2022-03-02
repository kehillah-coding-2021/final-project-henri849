import tensorflow as tf
import json
import numpy as np
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

ans = open("output.txt","r")
ans = json.loads(ans.read())

[[x_train, x_test], [y_train, y_test]] = ans
x_train = x_train[:5000]
x_test = x_test[:1000]
y_train = y_train[:5000]
y_test = y_test[:1000]

x_train = tf.ragged.constant(x_train).to_tensor()
x_test = tf.ragged.constant(x_test).to_tensor()
y_train = tf.ragged.constant(y_train).to_tensor()
y_test = tf.ragged.constant(y_test).to_tensor()
print(x_train)
model = tf.keras.Sequential([
    tf.keras.layers.Input(252,),
#     tf.keras.layers.Input(252),
    tf.keras.layers.Dense(125, activation='relu'),
    tf.keras.layers.Dense(5, activation='tanh'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200,batch_size=50)
model.evaluate(x_test,  y_test, verbose=2)
# tf.keras.models.save_model(model,"models2")
#model.save("models2")

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()


# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)