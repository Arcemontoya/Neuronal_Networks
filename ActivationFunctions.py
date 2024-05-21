import numpy as np

class Activation_ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        #Calcular valores de input
        self.output = 1.0 / (1.0 + np.exp(-inputs))


class Activation_Tanh:

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        # Calcular el gradiente de la función de activación tangente hiperbólica (Tanh)
        self.dinputs = dvalues * (1 - self.output ** 2)


class Activation_LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(self.alpha * inputs, inputs)

    def backward(self, dvalues):
        self.dinputs = np.where(self.inputs <= 0, self.alpha * dvalues, dvalues)


class Activation_ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))

    def backward(self, dvalues):
        self.dinputs = dvalues * np.where(self.inputs > 0, 1, self.alpha * np.exp(self.inputs))

class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs  # La salida es igual a la entrada

    def backward(self, dvalues):
        # El gradiente es igual al gradiente de la función de pérdida
        # Esto se hace para mantener la coherencia en la estructura de la red
        self.dinputs = dvalues
