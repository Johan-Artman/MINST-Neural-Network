import numpy as np
from keras.datasets import mnist
from keras import utils

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid
from losses import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict

def preprocess_data(x, y, limit):
    # Get indices for all digits
    indices = []
    for digit in range(10):
        digit_indices = np.where(y == digit)[0][:limit]
        indices.append(digit_indices)
    all_indices = np.hstack(indices)
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = utils.to_categorical(y, 10)  # 10 classes
    y = y.reshape(len(y), 10, 1)
    return x, y

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),  # 10 output classes
    Sigmoid()
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# Save the trained weights
def save_network_weights(network, filename='trained_network.npz'):
    weights_dict = {}
    for i, layer in enumerate(network):
        if hasattr(layer, 'weights'):
            weights_dict[f'layer_{i}_weights'] = layer.weights
        if hasattr(layer, 'bias'):
            weights_dict[f'layer_{i}_bias'] = layer.bias
    np.savez(filename, **weights_dict)
    print(f"Network weights saved to {filename}")

# Save after training
save_network_weights(network)

# test
correct = 0
total = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    pred = np.argmax(output)
    true = np.argmax(y)
    if pred == true:
        correct += 1
    total += 1
    print(f"pred: {pred}, true: {true}")

print(f"\nAccuracy: {correct/total * 100:.2f}%")
