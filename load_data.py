import numpy as np
import matplotlib.pyplot as plt


def load_labels(file_name):
    with open(file_name, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        N_numbers = int.from_bytes(f.read(4), "big")

        training_labels = np.frombuffer(
            f.read(),
            dtype=np.uint8
        ).reshape(N_numbers)

    return training_labels

def one_hot(label):
    return np.eye(10)[label]

def load_data(file_name):
    with open(file_name, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        N_numbers = int.from_bytes(f.read(4), "big")
        N_rows = int.from_bytes(f.read(4), "big")
        N_cols = int.from_bytes(f.read(4), "big")

        training_data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        ).reshape(N_numbers, N_rows, N_cols)

    return training_data


# training_inputs = load_data("mnist_data/t10k-images.idx3-ubyte")
# training_labels = load_labels("mnist_data/t10k-labels.idx1-ubyte")
# training_targets = create_target_from_label(training_labels)


# dim = 5
# fig, axs = plt.subplots(dim, dim)
# for x in range(dim):
#     for y in range(dim):
#         print(training_labels[x + dim * y], training_targets[x + dim * y])
#         axs[x, y].set_title(training_labels[x + dim * y])
#         axs[x, y].imshow(training_inputs[x + dim * y])

# plt.show()