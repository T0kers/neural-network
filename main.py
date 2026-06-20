import numpy as np
import matplotlib.pyplot as plt
import nn
import load_data

# data: https://web.archive.org/web/20200430193701/http://yann.lecun.com/exdb/mnist/

class NeuralNetwork(nn.Module):
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

        self.network = nn.Sequential([
            nn.Linear(28 * 28, 100),
            nn.Sigmoid(),
            nn.Linear(100, 10),
            nn.Softmax(True),
        ])
    
    def forward(self, x):
        return self.network.forward(x)

    def backward(self, grad, learning_rate):
        return self.network.backward(grad, learning_rate)
    
    def train(self, learning_rate, training_inputs, training_targets, N_epoch=1000):
        epoch_index = np.arange(N_epoch)
        epoch_losses = np.zeros(N_epoch)

        losses = np.zeros(len(training_inputs))
        training_inputs = np.reshape(training_inputs, (training_inputs.shape[0], np.prod(training_inputs.shape[1:])))
        for epoch in range(N_epoch):
            for i, (x, target) in enumerate(zip(training_inputs, training_targets)):
                prediction = self.forward(x)
                loss = self.loss_fn.eval(prediction, target)
                losses[i] = loss
                grad = self.loss_fn.grad(prediction, target)
                self.backward(grad, learning_rate)
            epoch_loss = np.mean(losses)
            epoch_losses[epoch] = epoch_loss
            print(f"Epoch {epoch}:", epoch_loss)
        return epoch_index, epoch_losses

training_inputs = load_data.load_data("mnist_data/t10k-images.idx3-ubyte")
training_labels = load_data.load_labels("mnist_data/t10k-labels.idx1-ubyte")
training_targets = load_data.one_hot(training_labels)

net = NeuralNetwork(nn.CrossEntropyLoss(True))
indexes, losses = net.train(0.001, training_inputs, training_targets)
print(net.network.layers[0].weights)
print(net.network.layers[0].biases)

plt.plot(indexes, losses, ",")
plt.show()
