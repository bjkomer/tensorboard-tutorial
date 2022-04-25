import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector
from datetime import datetime
from utils import tensorboard_activation_projector

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

log_dir = os.path.join(
    "logs/embedding_projector_example",
    datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
)

metadata_file = os.path.join(log_dir, "metadata.tsv")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    update_freq="epoch",
    profile_batch=2,
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

use_pytorch = False

if use_pytorch:
    # pytorch has a straightforward interface for tensorboard embeddings
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile  # required for the pytorch tensorboard interface to work
    from torch.utils.tensorboard import SummaryWriter
    # Get activations
    # print(list(map(lambda x: x.name, model.layers)))  # layer names, for getting activations
    dense_output = tf.keras.Model(
        model.input,
        model.get_layer('dense').output
    )

    activations = dense_output(x_test)


    # Write to tensorboard with the pytorch api
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_embedding(activations.numpy(), y_test)
    writer.close()
else:
    # Custom helper function to create embeddings without pytorch
    tensorboard_activation_projector(
        log_dir=log_dir,
        model=model,
        layer_names=['dense', 'dense_1'],
        input_data=x_test,
        labels=y_test
    )

