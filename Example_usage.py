import Dense_Construction
from Dense_Construction import NeuralNetwork ,FullyConnectedLayer
import numpy as np

data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

input_size = data.shape[1]
hidden_size = 4
output_size = labels.shape[1]
learning_rate = 0.3
num_epochs = 1000

nn = NeuralNetwork()
nn.add_layer(FullyConnectedLayer(input_size, hidden_size, activation='tanh'))
nn.add_layer(FullyConnectedLayer(hidden_size, output_size, activation='sigmoid'))

nn.train(data, labels, num_epochs, learning_rate)


test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_predictions = nn.prediction(test_data)
print("Test Predictions:")
print(test_predictions)
