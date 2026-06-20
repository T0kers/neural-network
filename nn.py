import numpy as np

def _not_implemented(s, name):
    raise NotImplementedError(
        f"{type(s).__name__} must implement {name}()"
    )

class Module:
    def __init__(self):
        self.cache = None
        self.what_to_save = None

    def cache_value(self, value):
        self.cache = value

    def forward(self, x):
        _not_implemented(self, "forward")
    
    def backward(self, grad, learning_rate):
        _not_implemented(self, "backward")
    
    def what_to_save(self):
        return None

    def save(self, path):
        np.savez(path, **self.what_to_save())
    
    def load(self, path):
        data = np.load(path)
        self.weights = data["weights"]
        self.biases = data["biases"]

class Linear(Module):
    def __init__(self, ins, outs):
        super().__init__()

        self.weights = np.random.uniform(low=-1, high=1, size=(outs, ins))
        self.biases = np.random.uniform(low=-1, high=1, size=outs)
    
    def forward(self, activation):
        self.cache_value(activation)
        return self.weights @ activation + self.biases

    def backward(self, grad, learning_rate):
        result = self.weights.T @ grad
        self.weights -= learning_rate * grad[:, np.newaxis] * self.cache[np.newaxis, :]
        self.biases -= learning_rate * grad
        return result

    def what_to_save(self):
        return {"weights": self.weights, "biases": self.biases}

class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z):
        self.cache_value(z)
        return np.maximum(0, z)
    
    def backward(self, grad, learning_rate):
        z = self.cache
        mask = z < 0
        z[mask] = 0
        z[~mask] = 1
        return z * grad

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z):
        a = 1 / (1 + np.exp(-z))
        self.cache_value(a)
        return a
    
    def backward(self, grad, learning_rate):
        return (self.cache * (1 - self.cache)) * grad

# https://parasdahal.com/softmax-crossentropy/
class Softmax(Module):
    def __init__(self, using_cross_entropy_loss):
        super().__init__()
    
        if using_cross_entropy_loss:
            self.backward = lambda grad, learning_rate: grad
        else:
            self.backward = lambda grad, learning_rate: (self.cache[:, np.newaxis] * (np.eye(4) - self.cache)) @ grad
    
    def forward(self, z):
        expz = np.exp(z - np.max(z))
        a = expz / np.sum(expz)
        self.cache_value(a)
        return a
    
    # def backward(self, grad, learning_rate):
    #     return (self.cache[:, np.newaxis] * (np.eye(4) - self.cache)) @ grad

class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad

class LossFn:
    def __init__(self):
        pass
    
    def eval(self, prediction, target):
        pass

    def grad(self, prediction, target):
        pass

class QuadraticLoss(LossFn):
    def eval(self, prediction, target):
        return np.mean((prediction - target)**2)
    
    def grad(self, prediction, target):
        return 2 / np.size(prediction) * (prediction - target)


class CrossEntropyLoss(LossFn):
    def __init__(self, using_softmax_and_onehot):
        if using_softmax_and_onehot:
            self.grad = lambda prediction, one_hot_target: prediction - one_hot_target
        else:
            raise "Not implemented"

    def eval(self, prediction, target):
        return -np.sum(target * np.log(prediction))
    