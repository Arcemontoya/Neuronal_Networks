import numpy as np

#Dense Layer
class Layer_Dense:

    def __init__(self, n_inptus, n_neurons,
                 weight_regularizer_L1=0, weight_regularizer_L2=0,
                 bias_regularizer_L1=0, bias_regularizer_L2=0):
        # Inicializa sesgos y pesos
        self.weights = 0.01 * np.random.randn(n_inptus, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Setea la fuerza de regularizacion
        self.weight_regularizer_L1 = weight_regularizer_L1
        self.weight_regularizer_L2 = weight_regularizer_L2
        self.bias_regularizer_L1 = bias_regularizer_L1
        self.bias_regularizer_L2 = bias_regularizer_L2

    # Paso a la siguiente capa
    def forward(self, inputs):
        # Calcula el valor de las entradas, pesos y sesgos
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        #Gradientes en parametros
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims=True)

        # Gradiente en regularizacion

        #L1 en pesos

        if self.weight_regularizer_L1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_L1 * dL1

        #L2 en pesos
        if self.weight_regularizer_L2 > 0:
            self.dweights += 2 * self.weight_regularizer_L2 * self.weights

        #L1 en biases
        if self.bias_regularizer_L1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_L1 * dL1

        #L2 en biases
        if self.bias_regularizer_L2 > 0:
            self.dbiases += 2 * self.bias_regularizer_L2 * self.biases

        # Gradinete en valores
        self.dinputs = np.dot(dvalues, self.weights.T)