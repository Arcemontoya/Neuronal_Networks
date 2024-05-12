import numpy as np

class Activation_ReLU:

    def forward(self, inputs):
        #Calcular valores de input
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        #Copia del valor
        self.dinputs = dvalues.copy()

        #Gradiente cero donde los valores son negativos
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:

    def forward(self, inputs):
        self.inputs = inputs

        #obtención de valores no normalizados
        exp_values = np.exp(inputs - np.max(inputs, axis= 1, keepdims =True))

        #Normalizacion
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):

        #Crea un array no inicializado
        self.dinputs = np.empty_like(dvalues)

        #Enumerar salidas y gradientes
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):

            #Flatten output array
            single_output = single_output.reshape(-1, 1)
            #Calcular la salida de la matriz Jacobiana y la salida
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            #Calcular sample-wise gradient y agregarlo al arrey de gradientes de ejemplo
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Sigmoid:

    def forward(self, inputs):
        self.inputs = inputs
        #Calcular valores de input
        self.output = 1.0 / (1.0 + np.exp(-inputs))


