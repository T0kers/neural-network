import numpy as np
import matplotlib.pyplot as plt


    
def train(net, loss_fn, training_inputs, training_targets, learning_rate=0.001, N_epoch=1000):
    epoch_index = np.arange(N_epoch)
    epoch_losses = np.zeros(N_epoch)

    losses = np.zeros(len(training_inputs))
    training_inputs = np.reshape(training_inputs, (training_inputs.shape[0], np.prod(training_inputs.shape[1:])))
    for epoch in range(N_epoch):
        for i, (x, target) in enumerate(zip(training_inputs, training_targets)):
            prediction = net.forward(x)
            loss = loss_fn.eval(prediction, target)
            losses[i] = loss
            grad = loss_fn.grad(prediction, target)
            net.backward(grad, learning_rate)
        epoch_loss = np.mean(losses)
        epoch_losses[epoch] = epoch_loss
        print(f"Epoch {epoch}:", epoch_loss)
    return epoch_index, epoch_losses
