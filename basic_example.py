import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from datetime import datetime

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


log_dir = os.path.join(
    "logs/basic_example",
    datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,  # how often to record weight histograms
    write_graph=True,  # visualize the computation graph
    write_images=False,  # view the learned weights as an image
    update_freq="epoch",  # how often to update, every "batch" or "epoch"
    profile_batch=2,  # which batch to use to compute memory/speed profiling
    embeddings_freq=0,
    embeddings_metadata=None,
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback]
)
