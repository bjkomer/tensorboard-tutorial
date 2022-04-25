import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from datetime import datetime
from tensorboard.plugins.hparams import api as hp
import argparse

parser = argparse.ArgumentParser("Train MNIST model with different hyperparameters")

parser.add_argument("--units", type=int, default=512)
parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid"])
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "rmsprop"])
parser.add_argument("--lr", type=float, default=0.001)

args = parser.parse_args()

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Naming the log directory based on the parameters helps for manual filtering
log_dir = os.path.join(
    "logs/hyperparameter_example",
    f"{args.activation}",
    f"{args.optimizer}",
    f"{args.units}",
    f"{args.lr}",
    datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
)

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
    tf.keras.layers.Dense(args.units, activation=args.activation),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


if args.optimizer == "adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
elif args.optimizer == "rmsprop":
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr)
else:
    raise NotImplementedError

model.compile(
    optimizer=optimizer,
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

hparams = {
    "units": args.units,
    "activation": args.activation,
    "optimizer": args.optimizer,
    "lr": args.lr,
}


with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams(hparams)

