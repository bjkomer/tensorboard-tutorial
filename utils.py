import os
import io
import tensorflow as tf
import matplotlib.pyplot as plt


def tensorboard_activation_projector(log_dir, model, layer_names, input_data, labels):
    config_file_name = f"projector_config.pbtxt"

    n_samples = input_data.shape[0]

    if not os.path.exists(os.path.join(log_dir, "projector")):
        os.makedirs(os.path.join(log_dir, "projector"))

    if isinstance(labels, dict):
        # Multiple labels are given, so all need to be included
        label_names = sorted(labels.keys())  # sort for consistency across runs
        with open(os.path.join(log_dir, "projector/metadata.tsv"), "w") as f:
            f.write("\t".join(label_names) + "\n")  # write headers
            for i in range(n_samples):
                for label_name in label_names[:-1]:
                    f.write(f"{labels[label_name][i]}\t")
                else:
                    # Write the last value with a newline instead of a tab
                    f.write(f"{labels[label_names[-1]][i]}\n")
    else:
        # Only one label is given, so a header must not be used
        assert labels.shape[0] == n_samples
        with open(os.path.join(log_dir, "projector/metadata.tsv"), "w") as f:
            for i in range(n_samples):
                f.write(f"{labels[i]}\n")

    for layer_name in layer_names:

        # TODO: get all activations at once
        dense_output = tf.keras.Model(
            model.input,
            model.get_layer(layer_name).output
        )

        with open(os.path.join(log_dir, config_file_name), "a") as f:
            f.write('embeddings {\n')
            f.write(f'  tensor_name: "{layer_name}:projector"\n')
            f.write(f'  metadata_path: "projector/metadata.tsv"\n')
            f.write(f'  tensor_path: "projector/{layer_name}/tensors.tsv"\n')
            f.write('}\n')

        proj_path = os.path.join(log_dir, "projector", layer_name)
        if not os.path.exists(proj_path):
            os.makedirs(proj_path)

        activations = dense_output(input_data)

        tensors = activations.numpy().reshape((n_samples, -1))
        n_dimensions = tensors.shape[1]

        with open(os.path.join(proj_path, "tensors.tsv"), "w") as f:
            for i in range(n_samples):
                for j in range(n_dimensions-1):
                    f.write(f"{tensors[i, j]}\t")
                else:
                    # Write the last value with a newline instead of a tab
                    f.write(f"{tensors[i, -1]}\n")


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    https://www.tensorflow.org/tensorboard/image_summaries
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

