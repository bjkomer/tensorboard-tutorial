import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import plot_to_image

n_epochs = 25
image_size = int(np.sqrt(n_epochs))

# Some simple data to plot
images = np.zeros((n_epochs, image_size, image_size))
sine = np.sin(np.arange(n_epochs))
cosine = np.cos(np.arange(n_epochs))

log_dir = os.path.join(
    "logs/custom_metrics_example",
    datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
)

writer = tf.summary.create_file_writer(log_dir)

for epoch in range(n_epochs):
    images[epoch, epoch // 5, epoch % 5] = 1
    with writer.as_default():
        # Scalar value
        tf.summary.scalar("my_metric/sine", sine[epoch], step=epoch)
        tf.summary.scalar("my_metric/cosine", cosine[epoch], step=epoch)

        # Matplotlib figure
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.imshow(images[epoch, :, :])
        tf.summary.image("my_image", plot_to_image(fig), step=epoch)
        # remove figure from memory, prevent plotting with plt.show()
        plt.close(fig)
        del fig
