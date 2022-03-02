import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
import tempfile
import json

# Backend agnostic way to save/restore models
# _, keras_file = tempfile.mkstemp('.h5')
# print('Saving model to: ', keras_file)
# tf.keras.models.save_model(model, keras_file, include_optimizer=False)

# Load the serialized model
loaded_model = tf.keras.models.load_model("models2")

epochs = 4
end_step = np.ceil(1.0 * 10000 / 20).astype(np.int32) * epochs
print(end_step)

new_pruning_params = {
      'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=100)
}

new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
new_pruned_model.summary()

new_pruned_model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer='adam',
    metrics=['accuracy'])

# Add a pruning step callback to peg the pruning step to the optimizer's
# step. Also add a callback to add pruning summaries to tensorboard
logdir = tempfile.mkdtemp()
callbacks = [
    sparsity.UpdatePruningStep(),
    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
]
###########################################################
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
############################################

new_pruned_model.fit(x_train, y_train,
          batch_size=20,
          epochs=20,
          verbose=1,
          callbacks=callbacks,
          validation_data=(x_test, y_test))

score = new_pruned_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
final_model = sparsity.strip_pruning(new_pruned_model)
final_model.summary()
final_model.save("models3")
