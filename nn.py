import numpy as np
from abc import ABC, abstractmethod

class Module(ABC):
    def __init__(self):
        self.cache = None

    def cache_value(self, value):
        self.cache = value

    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, grad, learning_rate):
        pass
    
    # This can be overridden by classes that inherit Module
    def state_dict(self):
        return {}
    
    # This can be overridden by classes that inherit Module
    def load_state_dict(self, state):
        pass

    def save(self, path):
        np.savez(path, **self.state_dict())
    
    def load(self, path):
        with np.load(path) as data:
            self.load_state_dict(dict(data))

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

    def state_dict(self):
        return {"weights": self.weights, "biases": self.biases}
    
    def load_state_dict(self, state):
        self.weights = state["weights"]
        self.biases = state["biases"]

class ReLU(Module):
    def __init__(self, ins):
        super().__init__()

        self.cache = np.zeros(ins)
    
    def forward(self, z):
        np.maximum(0, z, out=self.cache)
        return self.cache
    
    def backward(self, grad, learning_rate):
        return (self.cache > 0) * grad

class Sigmoid(Module):
    def __init__(self, ins):
        super().__init__()

        self.cache = np.zeros(ins)
    
    def forward(self, z):
        np.reciprocal(1 + np.exp(-z), out=self.cache)
        return self.cache
    
    def backward(self, grad, learning_rate):
        return (self.cache * (1 - self.cache)) * grad

# https://parasdahal.com/softmax-crossentropy/
class Softmax(Module):
    def __init__(self, ins, using_cross_entropy_loss):
        super().__init__()
        
        self.cache = np.zeros(ins)
        self.using_cross_entropy_loss = using_cross_entropy_loss
    
    def forward(self, z):
        expz = np.exp(z - np.max(z))
        np.divide(expz, np.sum(expz), out=self.cache)
        return self.cache
    
    def backward(self, grad, learning_rate):
        if self.using_cross_entropy_loss:
            return grad

        n = len(self.cache)
        jacobian = self.cache[:, None] * (np.eye(n) - self.cache)
        return jacobian @ grad

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
    
    def state_dict(self):
        state = {}
        for i, layer in enumerate(self.layers):
            for key, value in layer.state_dict().items():
                state[f"layer{i}.{key}"] = value
        return state
    
    def load_state_dict(self, state):
        for i, layer in enumerate(self.layers):
            prefix = f"layer{i}."
            layer.load_state_dict({key[len(prefix):]: val for key, val in state.items() if key.startswith(prefix)})

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
    