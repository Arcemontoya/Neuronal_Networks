import numpy as np

#Dense Layer
class Layer_Dense:

    def __init__(self, n_inptus, n_neurons):
        # Inicializa sesgos y pesos
        self.weights = 0.01 * np.random.randn(n_inptus, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Paso a la siguiente capa
    def forward(self, inputs):
        # Calcula el valor de las entradas, pesos y sesgos
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #Gradientes en parametros
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims=True)
        # Gradiente en valores
        self.dinputs = np.dot(dvalues, self.weights.T)