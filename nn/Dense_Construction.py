import numpy as np 

def tanh(x):
    return (np.exp(x)- np.exp(-x))/(np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    return 1 - np.tanh(x)**2


def softmax(x):
    e_x=np.exp(x- np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    if (x > 0).all():
        return 1
    else:
        return 0

def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros((1, output_size))
    
    def forward(self, x):
        self.input = x
        self.output_input = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'relu':
            self.output = relu(self.output_input)
            self.derivative = relu_derivative
        elif self.activation == 'sigmoid':
            self.output = sigmoid(self.output_input)
            self.derivative = sigmoid_derivative
        elif self.activation == 'tanh':
            self.output = tanh(self.output_input)
            self.derivative = tanh_derivative
        
        return self.output
    
    def backward(self, x, y, learning_rate):
        output_error = y - self.output
        output_delta = output_error * self.derivative(self.output_input)
        
        input_error = output_delta.dot(self.weights.T)
        input_delta = input_error * self.derivative(self.input)
        
        weight_gradients = self.input.T.dot(output_delta)
        bias_gradients = np.sum(output_delta, axis=0)
        
        self.weights += learning_rate * weight_gradients
        self.bias += learning_rate * bias_gradients
        
        return input_delta
    
    def prediction(self, x):
        self.input = x
        self.output_input = np.dot(x, self.weights) + self.bias
        
        if self.activation == 'relu':
            self.output = relu(self.output_input)
            self.derivative = relu_derivative
        elif self.activation == 'sigmoid':
            self.output = sigmoid(self.output_input)
            self.derivative = sigmoid_derivative
        elif self.activation == 'tanh':
            self.output = tanh(self.output_input)
            self.derivative = tanh_derivative
        
        return self.output
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y, learning_rate):
        for layer in reversed(self.layers):
            x = layer.backward(x, y, learning_rate)

    def train(self, x, y, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            prediction = self.forward(x)
            loss = mean_squared_error(y, prediction)
            self.backward(x, y, learning_rate)
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss}')
    



