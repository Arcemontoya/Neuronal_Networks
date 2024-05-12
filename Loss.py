import numpy as np

from ActivationFunctions import Activation_Softmax

# Realizar regularización L2
# Todos los pesos de la red al cuadrado
# Página 8 del resumen inicialización

# Clase común de pérdida
class Loss:

    #Calcula los datos y las perdidas dado a la salida del modelo y lo valores esperados
    def calculate(self, output, y):

        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    #Forward pass
    def forward(self, y_pred, y_true):

        # Numero de ejemplos en el lote
        samples = len(y_pred)

        # Evita la división entre 0
        # Recorta ambos lados para no arrastrar la media hacia ningún valor
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labes
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true, axis = 1
            )

            #Perdidas
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        #Calcula gradiente
        self.dinputs = -y_true / dvalues

        #Normalizar gradiente
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    #Forward pass
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)

        self.dinputs = dvalues.copy()

        self.dinputs[range(samples), y_true] -= 1

        self.dinputs = self.dinputs / samples