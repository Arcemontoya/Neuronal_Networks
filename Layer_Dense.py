import numpy as np

#Dense Layer
class Layer_Dense:

    def __int__(self, n_inptus, n_neurons):
        # Inicializa sesgos y pesos
        self.weights = 0.01 * np.random.randn(n_inptus, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Paso a la siguiente capa
    def forward(self, inputs):
        # Calcula el valor de las entradas, pesos y sesgos
        self.output = np.dot(input, self.weights) + self.biases